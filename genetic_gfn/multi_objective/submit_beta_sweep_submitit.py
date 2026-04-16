import argparse
import copy
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product

import yaml

try:
    import submitit
except Exception as e:  # pragma: no cover
    submitit = None
    _submitit_import_error = e

# Hard-coded lead SMILES for the `similarity` objective.
# Used by multi_objective/run.py via `--seed_mol` (Tanimoto similarity to this SMILES).
LEAD_SMILES = "CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3"


@dataclass(frozen=True)
class BetaCell:
    beta: float
    seed: int
    anti_target: str


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
    This file lives in: <REPO_ROOT>/multi_objective/submit_beta_sweep_submitit.py
    """
    script_path = _abs_path(__file__)
    multi_objective_dir = os.path.dirname(script_path)
    repo_root = os.path.dirname(multi_objective_dir)
    return repo_root, multi_objective_dir


def parse_beta_grid(hparam_config_path: str):
    """
    Accepts either:
    1) Simple list format used in this repo:
         beta: [5, 10, ...]
    2) W&B-sweep-ish format:
         parameters:
           beta:
             values: [5, 10, ...]
    """
    with open(hparam_config_path, "r") as f:
        hp = yaml.safe_load(f) or {}

    beta_vals = []

    # (1) Simple format: beta: [..]
    if isinstance(hp, dict) and "beta" in hp:
        beta_vals = hp.get("beta") or []

    # (2) Nested format: parameters.beta.values
    if not beta_vals:
        params = (hp.get("parameters") or {}) if isinstance(hp, dict) else {}
        beta_vals = ((params.get("beta") or {}).get("values")) or []

    if not beta_vals:
        raise ValueError(
            f"Invalid hparam config at {hparam_config_path}. Expected `beta: [...]` or `parameters.beta.values: [...]`."
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
    cell: BetaCell,
    *,
    out_dir: str,
    run_date_dir: str,
    max_oracle_calls: int,
    freq_log: int,
    target: str,
    oracle_url: str,
):
    """
    Runs ONE beta cell as a single job.

    - Uses multi_objective/run.py genetic_gfn
    - Uses --task simple (one job per seed)
    - Writes outputs under OUT_DIR/genetic_gfn/results/...
    """
    _, multi_objective_dir = _script_dirs()

    # Ensure relative paths in the repo work (targets, etc.)
    original_cwd = os.getcwd()
    os.chdir(multi_objective_dir)
    try:
        results_root = os.path.join(out_dir, "genetic_gfn", "results")
        # Flat output directory (filenames encode settings), grouped by date.
        sweep_out_dir = os.path.join(results_root, "beta_sweep", run_date_dir)
        os.makedirs(sweep_out_dir, exist_ok=True)

        base_config = os.path.join(multi_objective_dir, "genetic_gfn", "hparams_default.yaml")
        # Selectivity run name includes anti-target.
        run_name = f"{target}_anti{cell.anti_target}_beta{cell.beta}_seed{cell.seed}"
        # IMPORTANT: per-job output dir so molecules.csv/results don't overwrite across jobs.
        cell_out_dir = os.path.join(sweep_out_dir, run_name)
        os.makedirs(cell_out_dir, exist_ok=True)

        cfg_out = os.path.join(cell_out_dir, f"hparams_{run_name}.yaml")
        write_config(base_config, cfg_out, beta=cell.beta)

        cmd = [
            sys.executable,
            "-u",
            "run.py",
            "genetic_gfn",
            "--objectives",
            f"qed,sa,{target}",
            "--anti_target",
            str(cell.anti_target),
            "--alpha_vector",
            "1,1,1,1",
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
            "--oracle_url",
            str(oracle_url),
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
    parser.add_argument("--target", type=str, default="6nzp")
    parser.add_argument("--hparam_config", type=str, default="genetic_gfn/hparams_tune_simple.yaml")
    parser.add_argument("--max_oracle_calls", type=int, default=3000)
    parser.add_argument("--freq_log", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--oracle_url",
        type=str,
        required=True,
        help="Docking oracle base URL (e.g. 127.0.0.1:5456 or http://127.0.0.1:5456).",
    )
    parser.add_argument(
        "--max_betas",
        type=int,
        default=3,
        help="Use only the first N beta values from hparam_config (default: 3). "
             "With 3 anti-targets this yields 9 jobs for one seed.",
    )

    # Executor params
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--partition", type=str, default="a100")
    parser.add_argument("--cpus_per_task", type=int, default=12)
    parser.add_argument("--mem_gb", type=int, default=32)
    parser.add_argument("--timeout_min", type=int, default=48 * 60)
    parser.add_argument("--slurm_array_parallelism", type=int, default=50)
    parser.add_argument("--job_name", type=str, default="genetic_gfn_beta_sweep")
    parser.add_argument("--direct", action="store_true", default=False, help="Run with submitit LocalExecutor")

    args = parser.parse_args()

    out_dir = os.environ.get("OUT_DIR")
    if not out_dir:
        raise ValueError("OUT_DIR environment variable is required (root for results + submitit logs).")
    out_dir = _abs_path(out_dir)

    repo_root, multi_objective_dir = _script_dirs()

    # Make hparam_config absolute (relative to multi_objective/)
    hparam_config_path = args.hparam_config
    if not os.path.isabs(hparam_config_path):
        hparam_config_path = os.path.join(multi_objective_dir, hparam_config_path)

    beta_vals = parse_beta_grid(hparam_config_path)
    if args.max_betas is not None:
        beta_vals = beta_vals[: max(0, int(args.max_betas))]
    seeds = [int(s) for s in args.seeds]
    # Selectivity anti-targets for 6nzp.
    anti_targets = ["7uyt", "5ut5", "7uyw"] if str(args.target).lower() == "6nzp" else []
    if str(args.target).lower() == "6nzp" and not anti_targets:
        raise ValueError("6nzp target requires anti-targets but none were configured.")
    if str(args.target).lower() != "6nzp":
        raise ValueError("This submitit sweep script is configured for the 6nzp selectivity task only.")

    cells = [
        BetaCell(beta=b, seed=s, anti_target=at)
        for (b, s, at) in product(beta_vals, seeds, anti_targets)
    ]

    # Date folder to group results by run time
    run_date_dir = time.strftime("%Y-%m-%d_%H%M%S")

    # Submitit folder (logs) rooted at OUT_DIR
    ts = time.strftime("%Y%m%d_%H%M%S")
    submitit_root = os.path.join(out_dir, "genetic_gfn", "slurm_jobs", "submitit", "multi_objective_beta_sweep", ts)
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
    print("TARGET:", args.target)
    print("ANTI_TARGETS:", anti_targets)
    print("BETAS:", beta_vals)
    print("SEEDS:", seeds)
    print("NUM_JOBS:", len(cells))
    print("RUN_DATE_DIR:", run_date_dir)

    jobs = []
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
                target=args.target,
                oracle_url=args.oracle_url,
            )
            # NOTE: submitit forbids accessing job.job_id inside batch() context.
            submitted.append((job, cell))

    for job, cell in submitted:
        print("submitted:", job.job_id, cell)
        jobs.append(job)

    print(f"Submitted {len(jobs)} jobs.")


if __name__ == "__main__":
    main()

