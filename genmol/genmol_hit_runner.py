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
    get_log_dir
)


def prepare_hparam_config(original_config_dict, hparam_config_path):
    with open(hparam_config_path, "r") as f:
        hparam_config = yaml.safe_load(f)

    # Get all keys and their values
    keys = list(hparam_config.keys())
    values = [hparam_config[key] for key in keys]
    
    # Generate all possible combinations using Cartesian product
    config_dicts = []
    for combination in itertools.product(*values):
        config_dict = original_config_dict.copy()
        for key, value in zip(keys, combination):
            config_dict[key] = value
        config_dicts.append(config_dict)
    
    return config_dicts

def run_hit(cfg_path):
    """Run one GenMol hit job; GPU visibility from scheduler / environment."""
    cfg_abs = os.path.abspath(cfg_path)
    with open(cfg_abs, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["device"] = "cuda:0"
    temp_cfg = cfg_abs + ".tmp"
    with open(temp_cfg, "w") as f:
        yaml.safe_dump(cfg, f)

    repo = os.environ.get("PROJECT_ROOT", project_root)
    hit_run_py = os.path.abspath(os.path.join(repo, "genmol", "scripts", "exps", "hit", "run.py"))
    genmol_hit_command = [
        sys.executable,
        hit_run_py,
        "--config_file",
        temp_cfg,
    ]

    genmol_hit_process = subprocess.Popen(genmol_hit_command, env=os.environ.copy())
    genmol_hit_process.wait()
    
    # Cleanup
    if os.path.exists(temp_cfg):
        os.remove(temp_cfg)
    
    return genmol_hit_process.returncode

def run_hits(cfg_paths, max_workers=None):
    """Run multiple seed and oracle pairs in parallel in a single job"""
    job_start = time.time()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Limit parallel workers to avoid excessive memory usage
    if max_workers is None:
        max_workers = min(len(cfg_paths), 5)  # Default to max 5 parallel workers
    
    logging.info(f"Starting {len(cfg_paths)} hits with {max_workers} parallel workers")
    
    def run_single_hit(cfg_path):
        """Helper function to run a single seed and oracle pair"""
        try:
            return_code = run_hit(cfg_path)
            if return_code == 0:
                return cfg_path, None
            else:
                error_msg = f"Exit code: {return_code}"
                logging.error(f"{cfg_path}: {error_msg}")
                return cfg_path, error_msg
        except Exception as e:
            error_msg = str(e)
            logging.error(f"{cfg_path}: {error_msg}")
            return cfg_path, error_msg
    
    # Run hits in parallel using ThreadPoolExecutor with limited workers
    delay = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cfg = {}
        for i, cfg_path in enumerate(cfg_paths):
            if i > 0:
                time.sleep(delay)
            future = executor.submit(run_single_hit, cfg_path)
            future_to_cfg[future] = cfg_path
        
        results = []
        for future in as_completed(future_to_cfg):
            cfg_path, error = future.result()
            results.append((cfg_path, error))
    
    # Report summary
    job_time = time.time() - job_start
    successful = sum(1 for _, error in results if error is None)
    failed = len(results) - successful
    
    logging.info(f"Complete: {successful} successful, {failed} failed ({job_time/60:.1f}min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        type=str
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=False,
        type=int
    )
    parser.add_argument(
        "--oracle_name",
        nargs="+",
        required=False,
        type=str
    )
    parser.add_argument(
        "--oracle_url",
        required=False,
        type=str
    )
    parser.add_argument(
        "--max_oracle_calls",
        required=True,
        type=int
    )
    parser.add_argument(
        "--pool",
        type=str,
        choices=['scored', 'random'],
        required=False,
        default='random'
    )
    parser.add_argument(
        "--reward",
        type=str,
        choices=['hit', 'original', 'geam'],
        required=False,
        default='original'
    )
    parser.add_argument(
        "--hparam_config",
        type=str,
        required=False,
        default=None
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        required=False,
        default=15,
        help="Maximum number of parallel workers for running hits (default: min(5, num_configs))"
    )
    parser.add_argument(
        "--search_range",
        type=int,
        nargs="+",
        required=False,
        default=None
    )
    args = parser.parse_args()

    # Validate required arguments
    if args.oracle_name is None:
        raise ValueError("--oracle_name is required")
    if args.seeds is None:
        raise ValueError("--seeds is required")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    job_dir = get_job_dir(args.hparam_config is not None, cat=f"genmol-hit")
    logging.info("GPU selection: environment / SLURM; each subprocess uses cuda:0 in config.")

    root_dir = os.path.join(os.environ["PROJECT_ROOT"], "genmol")
    with open(os.path.join(root_dir, args.config_file), "r") as f:
        orig_config_dict = yaml.safe_load(f)
    if args.hparam_config is not None:
        config_dicts = prepare_hparam_config(orig_config_dict, os.path.join(root_dir, args.hparam_config))
    else:
        config_dicts = [orig_config_dict]

    
    all_cfg_paths = []

    if args.search_range is not None:
        config_dicts = config_dicts[args.search_range[0]:args.search_range[1]]
    for config_dict in config_dicts:
        # create log dirs
        model_name = args.config_file.split("/")[-1].split(".")[0]
        log_dir = get_log_dir(
            method=f"genetic-genmol",
            model_name=model_name,
            exp_name="exp",
            suffix="-hparam" if args.hparam_config else ""
        )
        os.makedirs(log_dir, exist_ok=True)

        for oracle in args.oracle_name:
            oracle_log_dir = os.path.join(log_dir, oracle)
            os.makedirs(oracle_log_dir, exist_ok=True)

            # Collect all config paths for this (hparam_combo, oracle) pair
            for seed in args.seeds:
                seed_log_dir = os.path.join(oracle_log_dir, f"seed-{seed}")
                os.makedirs(seed_log_dir, exist_ok=True)

                # Create a deep copy to avoid modifying the original
                seed_config_dict = config_dict.copy()
                
                # Update config with run-specific settings
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
                if seed_config_dict["reward"] == 'hit':
                    if oracle in ["jnk3", "drd2", "gsk3b"]:
                        seed_config_dict["task_name"] = f"hit.pmo"
                    else:
                        seed_config_dict["task_name"] = f"hit.{oracle}"
                if oracle in ["jnk3", "drd2", "gsk3b"]:
                    seed_config_dict["pmo-task"] = True
                else:
                    seed_config_dict["pmo-task"] = False
                
                seed_config_dict["device"] = "cuda:0"

                cfg_file = os.path.join(oracle_log_dir, f"config-{seed}.yaml")
                with open(cfg_file, "w") as f:
                    yaml.safe_dump(seed_config_dict, f)
                all_cfg_paths.append(cfg_file)

    total_workers = args.max_workers
    logging.info(f"Running {len(all_cfg_paths)} jobs with up to {total_workers} parallel workers")
    
    run_hits(cfg_paths=all_cfg_paths, max_workers=total_workers)
    logging.info(f"Completed {len(all_cfg_paths)} jobs")    