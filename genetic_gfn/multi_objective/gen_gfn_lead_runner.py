import argparse
import os
import sys
import subprocess
import itertools
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add repo root to path for utils imports (utils/experiment_utils.py lives at repo root)
# This file is at: <REPO_ROOT>/GeneticGFN/multi_objective/gen_gfn_lead_runner.py
repo_root = os.environ.get("PROJECT_ROOT")
if not repo_root:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

from utils.experiment_utils import (
    get_job_dir,
    get_log_dir
)

# Selectivity configuration: target -> list of anti-target docking receptors.
TARGET_TO_ANTI_TARGETS = {
    "6nzp": ["7uyw"],
}

# Hardcoded once from: GeneticGFN/multi_objective/genetic_gfn/lead/actives.csv
# Mapping: target -> list of seed_mol SMILES to run as similarity lead molecules.
TARGET_TO_SEED_MOLS = {
    "parp1": [
        "CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3",
        "COc1[nH]c3cccc2C(=O)NCCc1c23",
        "O/N=C/c1cn3CCNC(=O)c2cccc1c23",
    ],
    "fa7": [
        "CC(C)CCN(Cc2ccc1ccc(C(N)=N)cc1c2)C(=O)c3cccc4ccccc34",
        "N[C@H](Cc1ccccc1)C(=O)N2CCC[C@H]2C(=O)N[C@H](CCl)CCCN=C(N)N",
        "CC(C)Nc3ccc(c1cc(N)cc(C(O)=O)c1)n(CC(=O)NCc2ccc(C(N)=N)cc2)c3=O",
    ],
    "5ht1b": [
        "Cc1nc(-c2ccc(-c3ccc(C(=O)N4CCc5cc6c(cc54)[C@]4(CC[N@H+](C)CC4)CO6)cc3)c(C)c2)no1",
        "FC(F)(F)c1cccc(N2CC[NH2+]CC2)c1",
        "C1=CC2=NC=C(CCCN3CC[NH+](CCc4ccccc4)CC3)[C@H]2C=C1n1cnnc1",
    ],
    "braf": [
        "CCN(CC)CCNC(=O)c3cnn4c(c2cccc(NC(=O)Nc1ccc(Cl)c(C(F)(F)F)c1)c2)ccnc34",
        "FC(F)(F)c4cc(NC(=O)Nc3ccc(Oc2ccnc(C(=O)NCCN1CCOCC1)c2)cc3)ccc4Cl",
        "FC(F)(F)c4cc(NC(=O)Nc3ccc(Oc2ccnc(C(=O)Nc1cccnc1)c2)cc3)ccc4Cl",
    ],
    "jak2": [
        "OCCCCc2nc1ccccc1c4ncnc3[nH]cc2c34",
        "COC(=O)CC2Nc1ccccc1c3ccnc4[nH]cc2c34",
        "Oc5ccc(C2NC(=O)c1ccccc1c3ccnc4[nH]cc2c34)c(F)c5",
    ],
}

def _require_yaml():
    """
    Lazy import so `--help` works even if PyYAML isn't installed.
    We still require PyYAML for actual runs that read/write YAML.
    """
    try:
        import yaml  # type: ignore
        return yaml
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to run this script (for reading/writing configs). "
            "Install it with: `pip install pyyaml`"
        ) from e


def prepare_hparam_config(original_config_dict, hparam_config_path):
    yaml = _require_yaml()
    with open(hparam_config_path, "r") as f:
        hparam_config = yaml.safe_load(f)

    hparam_config = hparam_config or {}
    keys = list(hparam_config.keys())
    values = [hparam_config[key] if isinstance(hparam_config[key], list) else [hparam_config[key]] for key in keys]

    config_dicts = []
    for combination in itertools.product(*values):
        config_dict = original_config_dict.copy()
        for key, value in zip(keys, combination):
            config_dict[key] = value
        config_dicts.append(config_dict)

    return config_dicts


def _load_seed_mol_map(seed_mol_map_path: str):
    """
    Load a mapping of target -> seed_mol SMILES list.
    Expected format: YAML/JSON dict like:
      {parp1: ["CC...", "CC...", "CC..."], fa7: ["...", "...", "..."]}
    (A single string value is also accepted and will be treated as a 1-item list.)
    """
    if not seed_mol_map_path:
        return {}
    yaml = _require_yaml()
    with open(seed_mol_map_path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"--seed_mol_map must be a mapping (dict), got: {type(data)}")
    # normalize keys/values to list[str]
    out = {}
    for k, v in data.items():
        if v is None:
            continue
        key = str(k)
        if isinstance(v, (list, tuple)):
            vals = [str(x).strip() for x in v if str(x).strip()]
        else:
            vals = [str(v).strip()] if str(v).strip() else []
        out[key] = vals
    return out


def _parse_seed_mols_arg(seed_mol: str):
    """
    Parse --seed_mol which may be a single SMILES or a comma-separated list of SMILES.
    Returns list[str] (possibly empty).
    """
    s = (seed_mol or "").strip()
    if not s:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def _require_exactly_three_seed_mols(target: str, seed_mols: list, source: str):
    cleaned = [str(x).strip() for x in (seed_mols or []) if str(x).strip()]
    if len(cleaned) != 3:
        raise ValueError(
            f"Target '{target}' must have exactly 3 seed_mol SMILES from {source}, got {len(cleaned)}. "
            f"seed_mols={cleaned!r}"
        )
    return cleaned


def _get_lead_log_dir(*, method: str, model_name: str, suffix: str) -> str:
    """
    Lead-only results directory:
      $OUT_DIR/results/<method>/<model_name>/<YYYY-MM-DD><suffix>/lead/exp-<N>
    where <N> is the first available index under the lead/ folder.
    """
    out_dir = os.environ["OUT_DIR"]
    formatted_date_time = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(out_dir, "results", method, model_name, formatted_date_time) + (suffix or "")
    lead_root = os.path.join(date_dir, "lead")
    os.makedirs(lead_root, exist_ok=True)

    index = 0
    exp_suffix = f"exp-{index}"
    while os.path.exists(os.path.join(lead_root, exp_suffix)):
        index += 1
        exp_suffix = f"exp-{index}"
    return os.path.join(lead_root, exp_suffix)


def run_lead(cfg_path):
    """Run one GeneticGFN lead job; GPU visibility from scheduler / environment."""
    yaml = _require_yaml()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    gfn_lead_command = [
        "python3",
        "-u",
        f"{os.environ['PROJECT_ROOT']}/genetic_gfn/multi_objective/run.py",
        "genetic_gfn",
        "--task", "simple",
        "--seed", str(cfg["seed"]),
        "--objectives", str(cfg["objectives"]),
        "--alpha_vector", str(cfg.get("alpha_vector", "1,1,1,1")),
        "--max_oracle_calls", str(cfg["max_oracle_calls"]),
        "--freq_log", str(cfg.get("freq_log", 100)),
        "--output_dir", str(cfg["output_dir"]),
        "--run_name", str(cfg["run_name"]),
        "--config_default", str(cfg["config_default_path"]),
    ]

    # Lead molecule for similarity objective
    seed_mol = (cfg.get("seed_mol") or "").strip()
    if seed_mol:
        gfn_lead_command += ["--seed_mol", seed_mol]

    # Selectivity anti-target (required by run.py when objectives contain 6nzp)
    anti_target = (cfg.get("anti_target") or "").strip()
    if anti_target:
        gfn_lead_command += ["--anti_target", anti_target]

    # Pass through CPU parallelism (joblib pool size). run.py maps this to config["num_jobs"].
    if cfg.get("n_jobs") is not None:
        try:
            gfn_lead_command += ["--n_jobs", str(int(cfg["n_jobs"]))]
        except Exception:
            pass

    # Pass oracle_url explicitly (GeneticGFN reads it via args/config, not env vars)
    if cfg.get("oracle_url"):
        gfn_lead_command += ["--oracle_url", str(cfg["oracle_url"])]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    multi_obj_dir = os.path.join(os.environ["PROJECT_ROOT"], "genetic_gfn", "multi_objective")
    # Log stdout/stderr to per-run files so users can tail progress.
    run_dir = os.path.dirname(os.path.abspath(cfg_path))
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    run_name_safe = str(cfg.get("run_name") or "run").replace(os.sep, "_")
    stdout_path = os.path.join(logs_dir, f"{run_name_safe}.stdout.log")
    stderr_path = os.path.join(logs_dir, f"{run_name_safe}.stderr.log")

    with open(stdout_path, "w", buffering=1) as stdout_f, open(stderr_path, "w", buffering=1) as stderr_f:
        stdout_f.write(f"CWD: {multi_obj_dir}\n")
        stdout_f.write(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}\n")
        stdout_f.write("CMD: " + " ".join(gfn_lead_command) + "\n\n")
        stdout_f.flush()

        proc = subprocess.Popen(
            gfn_lead_command,
            env=env,
            cwd=multi_obj_dir,
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
        )
        proc.wait()
        return proc.returncode


def run_leads(cfg_paths, max_workers=None):
    """Run multiple lead jobs in parallel."""
    job_start = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if max_workers is None:
        max_workers = min(len(cfg_paths), 5)

    logging.info(f"Starting {len(cfg_paths)} lead runs with {max_workers} parallel workers")

    def run_single(cfg_path):
        try:
            rc = run_lead(cfg_path)
            if rc == 0:
                return cfg_path, None
            err = f"Exit code: {rc}"
            logging.error(f"{cfg_path}: {err}")
            return cfg_path, err
        except Exception as e:
            err = str(e)
            logging.error(f"{cfg_path}: {err}")
            return cfg_path, err

    delay = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cfg = {}
        for i, cfg_path in enumerate(cfg_paths):
            if i > 0:
                time.sleep(delay)
            future = executor.submit(run_single, cfg_path)
            future_to_cfg[future] = cfg_path

        results = []
        for future in as_completed(future_to_cfg):
            cfg_path, error = future.result()
            results.append((cfg_path, error))

    job_time = time.time() - job_start
    successful = sum(1 for _, error in results if error is None)
    failed = len(results) - successful
    logging.info(f"Complete: {successful} successful, {failed} failed ({job_time/60:.1f}min)")


if __name__ == "__main__":
    yaml = None  # lazy import later
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--targets", nargs="+", required=True, type=str)

    # Lead SMILES configuration:
    # - --seed_mol applies to ALL targets
    # - --seed_mol_map provides per-target seed_mol overrides (YAML mapping target->smiles)
    parser.add_argument(
        "--seed_mol",
        required=False,
        default="",
        type=str,
        help="Lead SMILES for similarity objective. Provide EXACTLY 3 comma-separated SMILES to run 3 leads per target, "
             "unless overridden by --seed_mol_map.",
    )
    parser.add_argument(
        "--seed_mol_map",
        required=False,
        default=None,
        type=str,
        help="YAML mapping of target->seed_mol SMILES (overrides --seed_mol per target).",
    )

    parser.add_argument("--oracle_url", required=False, type=str)
    parser.add_argument("--max_oracle_calls", required=True, type=int)
    parser.add_argument(
        "--n_jobs",
        required=False,
        default=-1,
        type=int,
        help="CPU workers passed to GeneticGFN/multi_objective/run.py --n_jobs. -1 means 'do not override YAML num_jobs'.",
    )
    parser.add_argument("--alpha_vector", required=False, default="1,1,1,1", type=str)
    parser.add_argument(
        "--objectives_prefix",
        required=False,
        default="qed,sa",
        type=str,
        help="Prefix for objectives (e.g., 'qed,sa' or 'qed,sa,similarity'). Target is appended automatically.",
    )
    parser.add_argument("--freq_log", required=False, default=100, type=int)
    parser.add_argument("--hparam_config", type=str, required=False, default=None)
    parser.add_argument(
        "--max_workers",
        type=int,
        required=False,
        default=15,
        help="Maximum parallel subprocesses (GPU from SLURM / environment; each run uses cuda:0).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve per-target lead SMILES map
    seed_mol_map = {}
    if args.seed_mol_map:
        seed_mol_map_path = args.seed_mol_map
        if not os.path.isabs(seed_mol_map_path):
            multi_obj_root = os.path.join(os.environ["PROJECT_ROOT"], "genetic_gfn", "multi_objective")
            seed_mol_map_path = os.path.join(multi_obj_root, seed_mol_map_path)
        seed_mol_map = _load_seed_mol_map(seed_mol_map_path)

    # Global seed mols (applied to all targets) must be exactly 3 if provided.
    global_seed_mols = _parse_seed_mols_arg(args.seed_mol)
    if global_seed_mols:
        global_seed_mols = _require_exactly_three_seed_mols("<all-targets>", global_seed_mols, source="--seed_mol")

    job_dir = get_job_dir(args.hparam_config is not None, cat="geneticgfn-lead")

    logging.info("GPU selection: environment / SLURM; meta-config uses cuda:0.")

    multi_obj_root = os.path.join(os.environ["PROJECT_ROOT"], "genetic_gfn", "multi_objective")
    base_cfg_path = args.config_file
    if not os.path.isabs(base_cfg_path):
        base_cfg_path = os.path.join(multi_obj_root, base_cfg_path)
    with open(base_cfg_path, "r") as f:
        yaml = _require_yaml()
        orig_config_dict = yaml.safe_load(f)
    orig_config_dict = orig_config_dict or {}

    hparam_path = None
    if args.hparam_config is not None:
        hparam_path = args.hparam_config
        if not os.path.isabs(hparam_path):
            hparam_path = os.path.join(multi_obj_root, args.hparam_config)
    config_dicts = prepare_hparam_config(orig_config_dict, hparam_path) if hparam_path else [orig_config_dict]

    all_cfg_paths = []

    for config_dict in config_dicts:
        model_name = os.path.basename(args.config_file).split(".")[0]
        log_dir = _get_lead_log_dir(
            method="genetic_gfn",
            model_name=model_name,
            suffix="-hparam" if args.hparam_config else "",
        )
        os.makedirs(log_dir, exist_ok=True)

        for target in args.targets:
            anti_targets = TARGET_TO_ANTI_TARGETS.get(str(target), [])
            anti_iter = anti_targets if anti_targets else [""]

            for anti_target in anti_iter:
                target_log_dir = os.path.join(log_dir, target)
                if anti_target:
                    target_log_dir = os.path.join(target_log_dir, f"anti-{anti_target}")
                os.makedirs(target_log_dir, exist_ok=True)

                # Pick lead SMILES for this target.
                # Priority:
                # 1) --seed_mol_map[target] (must provide 3)
                # 2) --seed_mol (must provide 3, applies to all targets)
                # 3) TARGET_TO_SEED_MOLS[target] (multiple, hardcoded from actives.csv)
                if target in seed_mol_map:
                    lead_smiles_list = _require_exactly_three_seed_mols(
                        target, seed_mol_map.get(target, []), source="--seed_mol_map"
                    )
                elif global_seed_mols:
                    lead_smiles_list = list(global_seed_mols)
                else:
                    if target not in TARGET_TO_SEED_MOLS:
                        raise ValueError(
                            f"Target '{target}' is missing from TARGET_TO_SEED_MOLS and no --seed_mol/--seed_mol_map was provided."
                        )
                    lead_smiles_list = _require_exactly_three_seed_mols(
                        target, TARGET_TO_SEED_MOLS.get(target, []), source="TARGET_TO_SEED_MOLS"
                    )

                # Use a nested loop so each target runs once per seed_mol.
                seedmol_iter = [(f"seedmol-{i}", smi) for i, smi in enumerate(lead_smiles_list)]

                for seedmol_tag, lead_smiles in seedmol_iter:
                    base_target_dir = os.path.join(target_log_dir, seedmol_tag) if seedmol_tag else target_log_dir
                    os.makedirs(base_target_dir, exist_ok=True)
                    # lead_smiles is guaranteed non-empty by _require_exactly_three_seed_mols()

                    for seed in args.seeds:
                        seed_log_dir = os.path.join(base_target_dir, f"seed-{seed}")
                        os.makedirs(seed_log_dir, exist_ok=True)

                        # 1) Write GeneticGFN hyperparam config_default YAML for this run
                        config_default_dict = dict(config_dict)
                        if args.oracle_url:
                            config_default_dict["oracle_url"] = args.oracle_url
                        config_default_path = os.path.join(seed_log_dir, "config_default.yaml")
                        with open(config_default_path, "w") as f:
                            yaml = _require_yaml()
                            yaml.safe_dump(config_default_dict, f, sort_keys=False)

                        # 2) Runner meta-config consumed by run_lead()
                        run_name = f"{target}_lead_task_seed{seed}"
                        if anti_target:
                            run_name = f"{run_name}_anti-{anti_target}"
                        if seedmol_tag:
                            run_name = f"{run_name}_{seedmol_tag}"
                        cfg_out = {
                            "seed": int(seed),
                            "target": str(target),
                            "anti_target": str(anti_target) if anti_target else "",
                            "seed_mol": (lead_smiles or "").strip(),
                            "objectives": f"{args.objectives_prefix},{target}",
                            "alpha_vector": str(args.alpha_vector),
                            "oracle_url": str(args.oracle_url) if args.oracle_url else "",
                            "max_oracle_calls": int(args.max_oracle_calls),
                            "freq_log": int(args.freq_log),
                            "output_dir": seed_log_dir,
                            "run_name": run_name,
                            "config_default_path": config_default_path,
                            "n_jobs": int(args.n_jobs) if args.n_jobs is not None else -1,
                            "device": "cuda:0",
                        }

                        cfg_file = os.path.join(seed_log_dir, "run_config.yaml")
                        with open(cfg_file, "w") as f:
                            yaml = _require_yaml()
                            yaml.safe_dump(cfg_out, f, sort_keys=False)
                        all_cfg_paths.append(cfg_file)

    total_workers = int(args.max_workers)
    logging.info(f"Running {len(all_cfg_paths)} jobs with up to {total_workers} parallel workers")
    run_leads(cfg_paths=all_cfg_paths, max_workers=total_workers)
    logging.info(f"Completed {len(all_cfg_paths)} jobs")

