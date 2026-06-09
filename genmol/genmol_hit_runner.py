import argparse
import os
import sys
import yaml
import subprocess
import itertools
import logging
import time
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
        config_dict = original_config_dict.copy()
        for key, value in zip(keys, combination):
            config_dict[key] = value
        config_dicts.append(config_dict)

    return config_dicts


def _device_for_job(job_index: int, devices: list[int]) -> int:
    return devices[job_index % len(devices)]


def run_hit(cfg_path: str, device: int) -> int:
    """Run one GenMol hit job; child sees cuda:0 on the chosen device."""
    cfg_abs = os.path.abspath(cfg_path)
    repo = os.environ.get("PROJECT_ROOT", project_root)
    hit_run_py = os.path.abspath(os.path.join(repo, "genmol", "scripts", "exps", "hit", "run.py"))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    logging.info("Launching %s on device %s", cfg_abs, device)
    genmol_hit_process = subprocess.Popen(
        [sys.executable, hit_run_py, "--config_file", cfg_abs],
        env=env,
    )
    genmol_hit_process.wait()
    return genmol_hit_process.returncode


def run_hits(cfg_paths, devices, max_workers=None):
    """Run hit jobs in parallel, round-robin across devices."""
    job_start = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if max_workers is None:
        max_workers = min(len(cfg_paths), 5)

    logging.info(
        "Starting %s hits with %s parallel workers on devices %s",
        len(cfg_paths),
        max_workers,
        devices,
    )

    def run_single_hit(job_index_cfg):
        job_index, cfg_path = job_index_cfg
        device = _device_for_job(job_index, devices)
        try:
            return_code = run_hit(cfg_path, device)
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
            future = executor.submit(run_single_hit, (i, cfg_path))
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
    parser.add_argument("--config_file", required=False, type=str)
    parser.add_argument("--seeds", nargs="+", required=False, type=int)
    parser.add_argument("--oracle_name", nargs="+", required=False, type=str)
    parser.add_argument("--oracle_url", required=False, type=str)
    parser.add_argument("--max_oracle_calls", required=True, type=int)
    parser.add_argument(
        "--pool",
        type=str,
        choices=["scored", "random"],
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
        help="Maximum number of parallel workers for running hits",
    )
    parser.add_argument("--search_range", type=int, nargs="+", required=False, default=None)
    args = parser.parse_args()

    if args.oracle_name is None:
        raise ValueError("--oracle_name is required")
    if args.seeds is None:
        raise ValueError("--seeds is required")

    devices = args.devices if args.devices is not None else [0]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    get_job_dir(args.hparam_config is not None, cat="genmol-hit")
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

    if args.search_range is not None:
        config_dicts = config_dicts[args.search_range[0] : args.search_range[1]]
    for config_dict in config_dicts:
        model_name = args.config_file.split("/")[-1].split(".")[0]
        log_dir = get_log_dir(
            method="genetic-genmol",
            model_name=model_name,
            exp_name="exp",
            suffix="-hparam" if args.hparam_config else "",
        )
        os.makedirs(log_dir, exist_ok=True)

        for oracle in args.oracle_name:
            oracle_log_dir = os.path.join(log_dir, oracle)
            os.makedirs(oracle_log_dir, exist_ok=True)

            for seed in args.seeds:
                seed_log_dir = os.path.join(oracle_log_dir, f"seed-{seed}")
                os.makedirs(seed_log_dir, exist_ok=True)

                seed_config_dict = config_dict.copy()
                seed_config_dict["model_path"] = os.path.join(
                    os.environ["PROJECT_ROOT"], "genmol", "model.ckpt"
                )
                seed_config_dict["seed"] = seed
                seed_config_dict["oracle_name"] = oracle
                seed_config_dict["output_dir"] = seed_log_dir
                seed_config_dict["max_oracle_calls"] = args.max_oracle_calls
                if args.oracle_url is not None:
                    seed_config_dict["oracle_url"] = args.oracle_url
                if not args.hparam_config:
                    seed_config_dict["pool"] = args.pool
                    seed_config_dict["reward"] = args.reward
                if seed_config_dict["reward"] == "hit":
                    if oracle in ["jnk3", "drd2", "gsk3b"]:
                        seed_config_dict["task_name"] = "hit.pmo"
                    else:
                        seed_config_dict["task_name"] = f"hit.{oracle}"
                seed_config_dict["pmo-task"] = oracle in ["jnk3", "drd2", "gsk3b"]

                cfg_file = os.path.join(oracle_log_dir, f"config-{seed}.yaml")
                with open(cfg_file, "w") as f:
                    yaml.safe_dump(seed_config_dict, f)
                all_cfg_paths.append(cfg_file)

    total_workers = args.max_workers
    logging.info("Running %s jobs with up to %s parallel workers", len(all_cfg_paths), total_workers)

    n_failed = run_hits(cfg_paths=all_cfg_paths, devices=devices, max_workers=total_workers)
    logging.info("Completed %s jobs", len(all_cfg_paths))
    sys.exit(1 if n_failed else 0)
