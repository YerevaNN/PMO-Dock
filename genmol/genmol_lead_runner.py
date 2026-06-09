import argparse
import os
import sys
import yaml
import subprocess
import itertools
import logging
import time
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for utils imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.experiment_utils import (
    get_job_dir,
    get_log_dir,
)


def prepare_hparam_config(original_config_dict, hparam_config_path):
    with open(hparam_config_path, "r") as f:
        hparam_config = yaml.safe_load(f)

    keys = list(hparam_config.keys())
    values = [hparam_config[key] for key in keys]

    config_dicts = []
    for combination in itertools.product(*values):
        config_dict = copy.deepcopy(original_config_dict)
        for key, value in zip(keys, combination):
            config_dict[key] = value
        config_dicts.append(config_dict)

    return config_dicts


def _device_for_job(job_index: int, devices: list[int]) -> int:
    return devices[job_index % len(devices)]


def run_lead(cfg_path: str, device: int) -> int:
    """Run one GenMol lead job; child sees cuda:0 on the chosen device."""
    cfg_abs = os.path.abspath(cfg_path)
    repo = os.environ.get("PROJECT_ROOT", project_root)
    lead_run_py = os.path.abspath(os.path.join(repo, "genmol", "scripts", "exps", "lead", "run.py"))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    logging.info("Launching %s on device %s", cfg_abs, device)
    genmol_lead_process = subprocess.Popen(
        [sys.executable, lead_run_py, "--config_file", cfg_abs],
        env=env,
    )
    genmol_lead_process.wait()
    return genmol_lead_process.returncode


def run_leads(cfg_paths, devices, max_workers=None):
    """Run lead jobs in parallel, round-robin across devices."""
    job_start = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if max_workers is None:
        max_workers = min(len(cfg_paths), 5)

    logging.info(
        "Starting %s lead experiments with %s parallel workers on devices %s",
        len(cfg_paths),
        max_workers,
        devices,
    )

    def run_single_lead(job_index_cfg):
        job_index, cfg_path = job_index_cfg
        device = _device_for_job(job_index, devices)
        try:
            return_code = run_lead(cfg_path, device)
            if return_code == 0:
                return cfg_path, None
            error_msg = f"Exit code: {return_code}"
            logging.error("%s: %s", cfg_path, error_msg)
            return cfg_path, error_msg
        except Exception as e:
            error_msg = str(e)
            logging.error("%s: %s", cfg_path, error_msg)
            return cfg_path, error_msg

    delay = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cfg = {}
        for i, cfg_path in enumerate(cfg_paths):
            if i > 0:
                time.sleep(delay)
            future = executor.submit(run_single_lead, (i, cfg_path))
            future_to_cfg[future] = cfg_path

        results = []
        for future in as_completed(future_to_cfg):
            cfg_path, error = future.result()
            results.append((cfg_path, error))

    job_time = time.time() - job_start
    successful = sum(1 for _, error in results if error is None)
    failed = len(results) - successful
    logging.info("Complete: %s successful, %s failed (%.1fmin)", successful, failed, job_time / 60)
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--oracle_names", nargs="+", required=True, type=str)
    parser.add_argument("--oracle_url", required=False, type=str, default=None)
    parser.add_argument("--max_oracle_calls", required=False, type=int, default=1000)
    parser.add_argument("-t", "--sim_thresholds", nargs="+", required=True, type=float)
    parser.add_argument("-i", "--start_mol_indices", nargs="+", required=True, type=int)
    parser.add_argument(
        "--pool",
        type=str,
        choices=["scored", "random", "seed"],
        required=False,
        default="random",
    )
    parser.add_argument(
        "--reward",
        type=str,
        choices=["hit", "original", "geam"],
        required=False,
        default="original",
    )
    parser.add_argument("--tox", action="store_true", default=False)
    parser.add_argument("--hparam_config", type=str, required=False, default=None)
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        required=False,
        default=None,
        help="GPU device ids for round-robin assignment (default: 0).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        required=False,
        default=None,
        help="Maximum number of parallel workers for running lead experiments",
    )
    parser.add_argument(
        "--num_completed",
        type=int,
        required=False,
        default=0,
        help="Number of completed jobs to skip",
    )
    args = parser.parse_args()

    devices = args.devices if args.devices is not None else [0]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    get_job_dir(args.hparam_config is not None, cat="genmol-lead")
    logging.info("GPU assignment: round-robin over devices %s", devices)

    root_dir = os.path.join(os.environ["PROJECT_ROOT"], "genmol")
    with open(os.path.join(root_dir, args.config_file), "r") as f:
        orig_config_dict = yaml.safe_load(f)
    if args.hparam_config is not None:
        config_dicts = prepare_hparam_config(
            orig_config_dict, os.path.join(root_dir, args.hparam_config)
        )
    else:
        config_dicts = [orig_config_dict]

    all_cfg_paths = []

    config_dicts = config_dicts[args.num_completed : 60]
    for config_dict in config_dicts:
        model_name = args.config_file.split("/")[-1].split(".")[0]
        log_dir = get_log_dir(
            method="genetic-genmol",
            model_name=model_name,
            exp_name="exp",
            suffix="-hparam" if args.hparam_config else "",
        )
        os.makedirs(log_dir, exist_ok=True)

        for oracle_name in args.oracle_names:
            oracle_log_dir = os.path.join(log_dir, oracle_name)
            os.makedirs(oracle_log_dir, exist_ok=True)

            for sim_thr in args.sim_thresholds:
                sim_thr_log_dir = os.path.join(oracle_log_dir, f"sim_thr-{sim_thr}")
                os.makedirs(sim_thr_log_dir, exist_ok=True)

                for start_mol_idx in args.start_mol_indices:
                    start_mol_idx_log_dir = os.path.join(
                        sim_thr_log_dir, f"start_mol_idx-{start_mol_idx}"
                    )
                    os.makedirs(start_mol_idx_log_dir, exist_ok=True)

                    for seed in args.seeds:
                        seed_log_dir = os.path.join(start_mol_idx_log_dir, f"seed-{seed}")
                        os.makedirs(seed_log_dir, exist_ok=True)

                        seed_config_dict = copy.deepcopy(config_dict)
                        seed_config_dict["model_path"] = os.path.join(
                            os.environ["PROJECT_ROOT"], "genmol", "model.ckpt"
                        )
                        seed_config_dict["seed"] = seed
                        seed_config_dict["oracle_name"] = oracle_name
                        seed_config_dict["log_dir"] = seed_log_dir
                        seed_config_dict["sim_thr"] = sim_thr
                        seed_config_dict["start_mol_idx"] = start_mol_idx
                        seed_config_dict["tox"] = args.tox
                        seed_config_dict["max_oracle_calls"] = args.max_oracle_calls
                        if not args.hparam_config:
                            seed_config_dict["pool"] = args.pool
                            seed_config_dict["reward"] = args.reward
                        if args.oracle_url is not None:
                            seed_config_dict["oracle_url"] = args.oracle_url

                        cfg_file = os.path.join(start_mol_idx_log_dir, f"config-{seed}.yaml")
                        with open(cfg_file, "w") as f:
                            yaml.safe_dump(seed_config_dict, f)
                        all_cfg_paths.append(cfg_file)

    total_workers = args.max_workers
    logging.info(
        "Running %s lead experiments with up to %s parallel workers",
        len(all_cfg_paths),
        total_workers,
    )

    n_failed = run_leads(cfg_paths=all_cfg_paths, devices=devices, max_workers=total_workers)
    logging.info("Completed %s lead experiments", len(all_cfg_paths))
    sys.exit(1 if n_failed else 0)
