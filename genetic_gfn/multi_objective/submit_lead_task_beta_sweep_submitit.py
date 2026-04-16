import argparse
import copy
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List

import yaml

try:
    import submitit
except Exception as e:  # pragma: no cover
    submitit = None
    _submitit_import_error = e


@dataclass(frozen=True)
class LeadBetaCell:
    target: str
    seed_smiles: str
    seed_mol_idx: int
    beta: float
    seed: int


def _require_submitit():
    if submitit is None:  # pragma: no cover
        raise ImportError(
            "submitit is not installed in this environment. Install it (e.g. `pip install submitit`) "
            f"or run on a cluster image that includes it. Original error: {_submitit_import_error}"
        )


def _abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _script_dirs():
    """
    This file lives in: <REPO_ROOT>/multi_objective/submit_lead_task_beta_sweep_submitit.py
    """
    script_path = _abs_path(__file__)
    multi_objective_dir = os.path.dirname(script_path)
    repo_root = os.path.dirname(multi_objective_dir)
    return repo_root, multi_objective_dir


def load_seed_mols_by_target(actives_csv_path: str) -> Dict[str, List[str]]:
    """
    Reads `benchmark/actives.csv` (default) and builds:
      { target: [seed_smiles, ...], ... }

    Only uses columns: `target`, `seed_smiles`.
    """
    by_target: Dict[str, List[str]] = {}
    with open(actives_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("target") or "").strip()
            s = (row.get("seed_smiles") or "").strip()
            if not t or not s:
                continue
            by_target.setdefault(t, []).append(s)
    return by_target


def parse_beta_grid(hparam_config_path: str):
    """
    Expects a yaml like multi_objective/genetic_gfn/hparams_tune.yaml:
      parameters:
        beta:
          values: [5, 10, ...]
    """
    with open(hparam_config_path, "r") as f:
        hp = yaml.safe_load(f) or {}

    params = (hp.get("parameters") or {}) or {}
    beta_vals = ((params.get("beta") or {}).get("values")) or []
    if not beta_vals:
        raise ValueError(
            f"Invalid hparam config at {hparam_config_path}. Expected parameters.beta.values."
        )
    return [float(x) for x in beta_vals]


def write_config(base_config_path: str, out_config_path: str, *, beta: float):
    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(cfg)
    cfg["beta"] = float(beta)

    os.makedirs(os.path.dirname(out_config_path), exist_ok=True)
    with open(out_config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_one(
    cell: LeadBetaCell,
    *,
    out_dir: str,
    run_date_dir: str,
    max_oracle_calls: int,
    freq_log: int,
):
    """
    Runs ONE lead-task cell as a single job.

    - Uses multi_objective/run.py genetic_gfn
    - Uses --task simple (one job per seed)
    - Uses --seed_mol <seed_smiles> which auto-adds `similarity` objective
    - Writes outputs under OUT_DIR/genetic_gfn/results/...
    """
    _, multi_objective_dir = _script_dirs()

    original_cwd = os.getcwd()
    os.chdir(multi_objective_dir)
    try:
        results_root = os.path.join(out_dir, "genetic_gfn", "results")
        cell_out_dir = os.path.join(results_root, "lead_task_beta_sweep", run_date_dir)
        os.makedirs(cell_out_dir, exist_ok=True)

        base_config = os.path.join(multi_objective_dir, "genetic_gfn", "hparams_default.yaml")
        run_name = f"{cell.target}_lead_beta{cell.beta}_seed{cell.seed}_mol{cell.seed_mol_idx}"
        cfg_out = os.path.join(cell_out_dir, f"hparams_{run_name}.yaml")
        write_config(base_config, cfg_out, beta=cell.beta)

        # NOTE:
        # - We pass objectives qed,sa,target.
        # - run.py will append `similarity` + an extra alpha weight automatically when --seed_mol is provided.
        cmd = [
            sys.executable,
            "-u",
            "run.py",
            "genetic_gfn",
            "--objectives",
            f"qed,sa,{cell.target}",
            "--alpha_vector",
            "1,1,1",
            "--seed_mol",
            cell.seed_smiles,
            "--max_oracle_calls",
            str(max_oracle_calls),
            "--task",
            "simple",
            "--seed",
            str(cell.seed),
            "--freq_log",
            str(freq_log),
            "--wandb",
            "disabled",
            "--run_name",
            run_name,
            "--output_dir",
            cell_out_dir,
            "--config_default",
            cfg_out,
        ]

        print("CWD:", os.getcwd())
        print("CMD:", " ".join(cmd))
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(cmd, check=False, env=env)
    finally:
        try:
            os.chdir(original_cwd)
        except Exception:
            pass


def main():
    _require_submitit()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actives_csv",
        type=str,
        default=None,
        help="Defaults to benchmark/oracles/actives.csv under PROJECT_ROOT.",
    )
    parser.add_argument("--targets", nargs="+", default=None, help="Optional subset of targets to run (default: all in actives.csv)")
    parser.add_argument("--hparam_config", type=str, default="genetic_gfn/hparams_tune.yaml")
    parser.add_argument("--max_oracle_calls", type=int, default=3000)
    parser.add_argument("--freq_log", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--max_seed_mols_per_target", type=int, default=0, help="0 = no limit; otherwise truncate seed_mols list per target.")

    # Executor params
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--partition", type=str, default="batch")
    parser.add_argument("--cpus_per_task", type=int, default=12)
    parser.add_argument("--mem_gb", type=int, default=32)
    parser.add_argument("--timeout_min", type=int, default=48 * 60)
    parser.add_argument("--slurm_array_parallelism", type=int, default=50)
    parser.add_argument("--job_name", type=str, default="genetic_gfn_lead_beta_sweep")
    parser.add_argument("--direct", action="store_true", default=False, help="Run with submitit LocalExecutor")

    args = parser.parse_args()

    out_dir = os.environ.get("OUT_DIR")
    if not out_dir:
        raise ValueError("OUT_DIR environment variable is required (root for results + submitit logs).")
    out_dir = _abs_path(out_dir)

    repo_root, multi_objective_dir = _script_dirs()

    # Make actives_csv + hparam_config absolute (relative to multi_objective/)
    if args.actives_csv is None:
        actives_csv_path = os.path.join(repo_root, "benchmark", "actives.csv")
    else:
        actives_csv_path = args.actives_csv
        if not os.path.isabs(actives_csv_path):
            actives_csv_path = os.path.join(multi_objective_dir, actives_csv_path)

    hparam_config_path = args.hparam_config
    if not os.path.isabs(hparam_config_path):
        hparam_config_path = os.path.join(multi_objective_dir, hparam_config_path)

    beta_vals = parse_beta_grid(hparam_config_path)
    seeds = [int(s) for s in args.seeds]

    seed_mols_by_target = load_seed_mols_by_target(actives_csv_path)
    if not seed_mols_by_target:
        raise ValueError(f"No seed molecules found in {actives_csv_path} (need columns: target, seed_smiles).")

    if args.targets:
        targets = [t for t in args.targets if t in seed_mols_by_target]
        missing = [t for t in args.targets if t not in seed_mols_by_target]
        if missing:
            print("WARNING: targets not found in actives.csv (skipping):", missing)
    else:
        targets = sorted(seed_mols_by_target.keys())

    # Optionally truncate per target
    if args.max_seed_mols_per_target and args.max_seed_mols_per_target > 0:
        for t in list(targets):
            seed_mols_by_target[t] = seed_mols_by_target[t][: args.max_seed_mols_per_target]

    # Build grid
    cells: List[LeadBetaCell] = []
    for target in targets:
        seed_mols = seed_mols_by_target.get(target, [])
        for seed_mol_idx, seed_smiles in enumerate(seed_mols):
            for beta, seed in product(beta_vals, seeds):
                cells.append(
                    LeadBetaCell(
                        target=target,
                        seed_smiles=seed_smiles,
                        seed_mol_idx=int(seed_mol_idx),
                        beta=float(beta),
                        seed=int(seed),
                    )
                )

    # Date folder to group results by run time
    run_date_dir = time.strftime("%Y-%m-%d_%H%M%S")

    # Submitit folder (logs) rooted at OUT_DIR
    ts = time.strftime("%Y%m%d_%H%M%S")
    submitit_root = os.path.join(out_dir, "genetic_gfn", "slurm_jobs", "submitit", "multi_objective_lead_beta_sweep", ts)
    os.makedirs(submitit_root, exist_ok=True)

    if args.direct:
        executor = submitit.LocalExecutor(folder=os.path.join(submitit_root, "%j"))
        executor.update_parameters(
            timeout_min=args.timeout_min,
            gpus_per_node=args.n_gpus,
            nodes=1,
            mem_gb=args.mem_gb,
            cpus_per_task=args.cpus_per_task,
        )
    else:
        executor = submitit.AutoExecutor(folder=os.path.join(submitit_root, "%j"))
        executor.update_parameters(
            slurm_job_name=args.job_name,
            timeout_min=args.timeout_min,
            slurm_array_parallelism=args.slurm_array_parallelism,
            gpus_per_node=args.n_gpus,
            nodes=1,
            mem_gb=args.mem_gb,
            cpus_per_task=args.cpus_per_task,
            slurm_additional_parameters={"partition": args.partition},
        )

    print("REPO_ROOT:", repo_root)
    print("MULTI_OBJECTIVE_DIR:", multi_objective_dir)
    print("OUT_DIR:", out_dir)
    print("SUBMITIT_LOG_ROOT:", submitit_root)
    print("ACTIVES_CSV:", actives_csv_path)
    print("HPARAM_CONFIG:", hparam_config_path)
    print("TARGETS:", targets)
    print("SEED_MOLS_BY_TARGET:", {t: len(seed_mols_by_target[t]) for t in targets})
    print("BETAS:", beta_vals)
    print("SEEDS:", seeds)
    print("NUM_JOBS:", len(cells))
    print("RUN_DATE_DIR:", run_date_dir)

    submitted = []
    with executor.batch():
        for cell in cells:
            job = executor.submit(
                run_one,
                cell=cell,
                out_dir=out_dir,
                run_date_dir=run_date_dir,
                max_oracle_calls=args.max_oracle_calls,
                freq_log=args.freq_log,
            )
            submitted.append((job, cell))

    for job, cell in submitted:
        print("submitted:", job.job_id, cell)

    print(f"Submitted {len(submitted)} jobs.")


if __name__ == "__main__":
    main()

