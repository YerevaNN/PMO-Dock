import argparse
import os
import sys
import yaml
import subprocess
import submitit
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
    """Run a single hit experiment"""
    genmol_hit_command = [
        "python3",
        f"{os.environ['PROJECT_ROOT']}/GenMol/scripts/exps/hit/run.py",
        "--config_file",
        cfg_path
    ]
    
    genmol_hit_process = subprocess.Popen(genmol_hit_command)
    genmol_hit_process.wait()
    
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
    delay = 30
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
        required=False
    )
    parser.add_argument(
        "--reward",
        type=str,
        choices=['hit', 'original', 'geam'],
        required=False
    )
    parser.add_argument(
        "--n_gpus",
        required=False,
        default=1,
        type=int
    )
    parser.add_argument(
        "--partition",
        required=False,
        type=str,
        default="batch"
    )
    parser.add_argument(
        "--direct",
        action="store_true"
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
    args = parser.parse_args()

    # Validate required arguments
    if args.oracle_name is None:
        raise ValueError("--oracle_name is required")
    if args.seeds is None:
        raise ValueError("--seeds is required")

    job_dir = get_job_dir(args.hparam_config is not None, cat=f"genmol-hit")
    

    if args.direct:
        executor = submitit.LocalExecutor(folder=f"{job_dir}/%j")
        n_gpus = args.n_gpus
        executor.update_parameters(
            timeout_min=10 * 60,
            gpus_per_node=n_gpus,
            nodes=1,
            mem_gb=160,  # Increased from 80GB to handle multiple parallel processes
            cpus_per_task=32
        )
    else:
        executor = submitit.AutoExecutor(folder=f"{job_dir}/%j")
        n_gpus = args.n_gpus
        slurm_additional_parameters = {}
        if args.partition is not None:
            slurm_additional_parameters["partition"] = args.partition

        executor.update_parameters(
        # name="llama-chem",
        slurm_job_name=os.environ["SLURM_JOB_NAME"],
        slurm_account=os.environ["SLURM_ACCOUNT"],
        timeout_min=10 * 60,  # Increased timeout since each job now runs 5 seeds
        slurm_array_parallelism=10,
        # slurm_gres=f"gpu:{n_gpus}",
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=160,  # Increased from 80GB to handle multiple parallel processes
        cpus_per_task=10,
        slurm_additional_parameters=slurm_additional_parameters,
    )

    jobs = []
    args_dict = vars(args)

    root_dir = os.path.join(os.environ["PROJECT_ROOT"], "GenMol")
    with open(os.path.join(root_dir, args.config_file), "r") as f:
        orig_config_dict = yaml.safe_load(f)
    if args.hparam_config is not None:
        config_dicts = prepare_hparam_config(orig_config_dict, os.path.join(root_dir, args.hparam_config))
    else:
        config_dicts = [orig_config_dict]

    
    # Prepare all job groups (hparam_combo, oracle) pairs
    all_job_groups = []
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

        # Group configs by (seed, oracle_name) - each group will be one job
        seed_config_paths = []
        for oracle in args.oracle_name:
            oracle_log_dir = os.path.join(log_dir, oracle)
            os.makedirs(oracle_log_dir, exist_ok=True)

            # Collect all config paths for this (hparam_combo, oracle) pair
            for device_ind, seed in enumerate(args.seeds):
                seed_log_dir = os.path.join(oracle_log_dir, f"seed-{seed}")
                os.makedirs(seed_log_dir, exist_ok=True)

                
                # Create a deep copy to avoid modifying the original
                seed_config_dict = config_dict.copy()
                
                # Update config with run-specific settings
                seed_config_dict["model_path"] = os.path.join(os.environ["PROJECT_ROOT"], "GenMol", "model.ckpt")
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
                # For local execution, use cuda:0 for all seeds if n_gpus=1
                if args.direct and n_gpus == 1:
                    seed_config_dict["device"] = "cuda:0"
                else:
                    seed_config_dict["device"] = f"cuda:{device_ind % n_gpus}"

                cfg_file = os.path.join(oracle_log_dir, f"config-{seed}.yaml")
                with open(cfg_file, "w") as f:
                    yaml.safe_dump(seed_config_dict, f)
                seed_config_paths.append(cfg_file)
            
        all_job_groups.append(seed_config_paths)

    # Submit all jobs
    if args.direct:
        for seed_config_paths in all_job_groups:
            job = executor.submit(run_hits, cfg_paths=seed_config_paths, max_workers=args.max_workers)
            job.result()
            jobs.append(job)
    else:
        for i, seed_config_paths in enumerate(all_job_groups):
            job = executor.submit(run_hits, cfg_paths=seed_config_paths, max_workers=args.max_workers)
            if (i + 1) % 100 == 0:
                logging.info(f"Submitted {i + 1}/{len(all_job_groups)} jobs")
            jobs.append(job)
        logging.info(f"Submitted {len(all_job_groups)} jobs")    