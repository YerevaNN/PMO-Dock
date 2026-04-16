import argparse
import os
import sys
import yaml
import subprocess
import submitit
import itertools
import copy
import pandas as pd
import time
import torch
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.experiment_utils import (
    get_job_dir,
    get_log_dir,
    generate_random_hash
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
        config_dict = copy.deepcopy(original_config_dict)
        for key, value in zip(keys, combination):
            if key == "sigma":
                config_dict["goal_directed_generation"]["reinforcement_learning"]["sigma"] = value
            elif key == "execute_hallucinated_memory":
                config_dict["goal_directed_generation"]["hallucinated_memory"]["execute_hallucinated_memory"] = value
            else:
                config_dict[key] = value
        config_dicts.append(config_dict)
    
    return config_dicts

def run_saturn_lead(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["device"] = "cuda:0"
    temp_cfg = cfg_path + ".tmp"
    with open(temp_cfg, "w") as f:
        yaml.safe_dump(cfg, f)

    repo = os.environ["PROJECT_ROOT"]
    saturn_command = [
        "python3",
        os.path.join(repo, "saturn", "saturn.py"),
        "--config",
        temp_cfg,
    ]

    print(f"cuda:0 (scheduler GPU): {' '.join(saturn_command)}")
    saturn_process = subprocess.Popen(saturn_command, env=os.environ.copy())
    saturn_process.wait()
    
    # Cleanup
    if os.path.exists(temp_cfg):
        os.remove(temp_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
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
        "--oracle_names",
        nargs="+",
        required=False,
        type=str
    )
    parser.add_argument(
        "--max_oracle_calls",
        required=True,
        type=int,                                                                           
        default=1000
    )
    parser.add_argument(
        "-i",
        "--start_mol_idx",
        required=False,
        type=int,
        nargs="+",
        default=[0, 1, 2]
    )
    parser.add_argument(
        "--fix_lead_smiles_in_buffer",
        action="store_true"
    )
    parser.add_argument(
        "-ga",
        "--execute_hallucinated_memory",
        action="store_true",
        required=False,
        default=None
    )
    parser.add_argument(
        "--sigma",
        required=False,
        type=float,
        default=128.0
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
        "--oracle_url",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "--jobs_per_gpu",
        type=int,
        required=False,
        default=1,
        help="Number of jobs to run in parallel on each GPU"
    )
    args = parser.parse_args()

    job_dir = get_job_dir(args.hparam_config is not None, cat=f"saturn-lead")
    print(f"job_dir: {job_dir}")
    

    if args.direct:
        executor = submitit.LocalExecutor(folder=f"{job_dir}/%j")
        # Detect all available GPUs when using LocalExecutor
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available_gpus > 0:
            n_gpus = available_gpus
            print(f"Detected {n_gpus} available GPU(s) for LocalExecutor")
        else:
            n_gpus = args.n_gpus
            print(f"No CUDA devices detected, using n_gpus={n_gpus} from argument")
        executor.update_parameters(
            timeout_min=10 * 60,
            gpus_per_node=n_gpus,
            nodes=1,
            mem_gb=80,
            cpus_per_task=32
        )
    else:
        executor = submitit.AutoExecutor(folder=f"{job_dir}/%j")
        n_gpus = args.n_gpus
        slurm_additional_parameters = {}
        if args.partition is not None:
            slurm_additional_parameters["partition"] = args.partition

        executor.update_parameters(
        slurm_job_name=os.environ["SLURM_JOB_NAME"],
        slurm_account=os.environ["SLURM_ACCOUNT"],
        timeout_min=2 * 60,
        slurm_array_parallelism=10,
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=80,
        cpus_per_task=20,
        slurm_additional_parameters=slurm_additional_parameters,
    )

    jobs = []
    args_dict = vars(args)
    with open(args.config, "r") as f:
        orig_config_dict = yaml.safe_load(f)
    if args.hparam_config is not None:
        config_dicts = prepare_hparam_config(orig_config_dict, args.hparam_config)
    else:
        config_dicts = [copy.deepcopy(orig_config_dict)]
    lead_smiles_df = pd.read_csv(os.path.join(os.environ["PROJECT_ROOT"], "benchmark", "actives.csv"))
    print(f"{len(config_dicts)} configs prepared to run")

    config_paths = []
    for config_dict_idx, config_dict in enumerate(config_dicts):
        model_name = args.config.split("/")[-1].split(".")[0]
        base_log_dir = get_log_dir(
            method=f"saturn",
            model_name=model_name,
            exp_name="exp",
            suffix=f"-hparam" if args.hparam_config else ""
        )
        os.makedirs(base_log_dir, exist_ok=True)
        log_dir = base_log_dir
        
        for i in args.start_mol_idx:
            start_idx_dir = os.path.join(log_dir, f"start_mol_idx-{i}")
            os.makedirs(start_idx_dir, exist_ok=True)

            for oracle_name in args.oracle_names:
                start_smiles = lead_smiles_df[lead_smiles_df["target"] == oracle_name]['smiles'].iloc[i]
                oracle_log_dir = os.path.join(start_idx_dir, oracle_name)
                os.makedirs(oracle_log_dir, exist_ok=True)

                for seed in args.seeds:
                    seed_log_dir = os.path.join(oracle_log_dir, f"seed-{seed}")
                    os.makedirs(seed_log_dir, exist_ok=True)

                    # Create a fresh copy of config for each seed
                    seed_config_dict = copy.deepcopy(config_dict)
                    experiment_id = generate_random_hash(8)
                    
                    # Update config with run-specific settings
                    seed_config_dict["seed"] = seed
                    seed_config_dict["oracle"]["components"][0]["specific_parameters"]["target"] = oracle_name
                    if args.oracle_url is not None:
                        seed_config_dict["oracle"]["components"][0]["specific_parameters"]["oracle_url"] = args.oracle_url
                    seed_config_dict["logging"]["logging_path"] = os.path.join(seed_log_dir, "results.log")
                    seed_config_dict["logging"]["model_checkpoints_dir"] = seed_log_dir
                    seed_config_dict["oracle"]["budget"] = args.max_oracle_calls
                    seed_config_dict["goal_directed_generation"]["experience_replay"]["lead_smiles_in_buffer"] = start_smiles
                    seed_config_dict["oracle"]["components"][0]["lead_smiles"] = start_smiles
                    seed_config_dict["goal_directed_generation"]["experience_replay"]["fix_lead_smiles_in_buffer"] = args.fix_lead_smiles_in_buffer
                    if args.execute_hallucinated_memory is not None:
                        seed_config_dict["goal_directed_generation"]["hallucinated_memory"]["execute_hallucinated_memory"] = args.execute_hallucinated_memory
                    seed_config_dict["device"] = "cuda:0"

                    cfg_file = os.path.join(oracle_log_dir, f"config-{seed}.yaml")
                    
                    # Normalize the path to ensure consistency
                    cfg_file = os.path.abspath(cfg_file)
                    
                    with open(cfg_file, "w") as f:
                        yaml.safe_dump(seed_config_dict, f)
                    config_paths.append(cfg_file)

    # Organize config paths by device for controlled parallel execution
    device_queues = {}
    for cfg_path in config_paths:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        device = cfg.get("device", "cuda:0")
        if device not in device_queues:
            device_queues[device] = []
        device_queues[device].append(cfg_path)

    if args.direct:
        # Track running jobs per device
        running_jobs = {device: [] for device in device_queues.keys()}
        all_jobs = []
        
        # Submit and manage jobs with parallelism control per device
        while any(len(queue) > 0 for queue in device_queues.values()) or any(len(jobs) > 0 for jobs in running_jobs.values()):
            for device in list(device_queues.keys()):
                # Remove completed jobs
                completed = [job for job in running_jobs[device] if job.done()]
                for job in completed:
                    try:
                        job.result()  # Check for errors
                        print(f"Job on {device} completed")
                    except Exception as e:
                        print(f"Job on {device} failed: {e}")
                    running_jobs[device].remove(job)
                
                # Submit new jobs if there's capacity
                while len(running_jobs[device]) < args.jobs_per_gpu and len(device_queues[device]) > 0:
                    cfg_path = device_queues[device].pop(0)
                    job = executor.submit(run_saturn_lead, cfg_path=cfg_path)
                    running_jobs[device].append(job)
                    all_jobs.append(job)
                    print(f"Submitted job on {device} ({len(running_jobs[device])}/{args.jobs_per_gpu}): {cfg_path}")
            
            # Small sleep to avoid busy waiting
            if any(len(jobs) > 0 for jobs in running_jobs.values()):
                time.sleep(0.1)
        
        # Wait for remaining jobs
        for job in all_jobs:
            if not job.done():
                job.result()
    else:
        with executor.batch():
            for cfg_path in config_paths:
                job = executor.submit(run_saturn_lead, cfg_path=cfg_path)
                print(job)
                jobs.append(job)    