import argparse
import copy
import os
import sys
import time
import yaml
import subprocess
import submitit
import itertools
import torch

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.experiment_utils import (
    get_job_dir,
    get_log_dir,
    generate_random_hash
)


def deep_merge(base_dict, update_dict):
    """
    Deeply merge update_dict into base_dict, preserving base_dict structure.
    """
    import copy
    result = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            # For lists, merge each element if it's a dict, otherwise replace
            if len(result[key]) > 0 and len(value) > 0:
                if isinstance(result[key][0], dict) and isinstance(value[0], dict):
                    # Merge list of dicts
                    merged_list = []
                    for i in range(max(len(result[key]), len(value))):
                        if i < len(result[key]) and i < len(value):
                            merged_list.append(deep_merge(result[key][i], value[i]))
                        elif i < len(result[key]):
                            merged_list.append(copy.deepcopy(result[key][i]))
                        else:
                            merged_list.append(copy.deepcopy(value[i]))
                    result[key] = merged_list
                else:
                    # For non-dict lists, replace
                    result[key] = copy.deepcopy(value)
            else:
                result[key] = copy.deepcopy(value)
        else:
            # For non-dict values or new keys, assign
            result[key] = copy.deepcopy(value)
    return result


def extract_hparam_combinations(hparam_config):
    """
    Extract all hparam combinations from nested structure.
    Returns a list of dicts, each representing one hparam path -> value mapping.
    """
    def _extract_paths(obj, prefix="", result=None):
        if result is None:
            result = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, list) and len(value) > 0:
                    # Check if it's a list of values (hparam) or list of dicts (nested)
                    if not isinstance(value[0], dict):
                        # This is a list of scalar values - this is an hparam
                        result.append((new_prefix, value))
                    else:
                        # List of dicts - recurse into each with index
                        for i, item in enumerate(value):
                            _extract_paths(item, f"{new_prefix}[{i}]", result)
                elif isinstance(value, dict):
                    # Nested dict - recurse
                    _extract_paths(value, new_prefix, result)
                else:
                    # Single value - wrap in list
                    result.append((new_prefix, [value]))
        return result
    
    return _extract_paths(hparam_config)


def set_nested_value(config_dict, path, value):
    """
    Set a value in a nested dict using a dot-separated path with optional list indices.
    E.g., "oracle.components[0].name" -> config_dict["oracle"]["components"][0]["name"] = value
    """
    import re
    # Split path into parts: keys and indices
    # Pattern matches: word characters (keys) or [digits] (indices)
    parts = re.findall(r'(\w+)|\[(\d+)\]', path)
    
    # Process parts into (type, value) tuples
    processed_parts = []
    for part in parts:
        if part[1]:  # It's an index [digits]
            processed_parts.append(('index', int(part[1])))
        else:  # It's a key
            processed_parts.append(('key', part[0]))
    
    # Navigate to the target location
    current = config_dict
    for i, (part_type, part_value) in enumerate(processed_parts[:-1]):
        if part_type == 'key':
            if part_value not in current:
                # Check if next part is an index
                if i + 1 < len(processed_parts) and processed_parts[i + 1][0] == 'index':
                    current[part_value] = []
                else:
                    current[part_value] = {}
            current = current[part_value]
        else:  # index
            idx = part_value
            while len(current) <= idx:
                # Check if next part is an index or key
                if i + 1 < len(processed_parts):
                    if processed_parts[i + 1][0] == 'index':
                        current.append([])
                    else:
                        current.append({})
                else:
                    current.append({})
            current = current[idx]
    
    # Set the final value
    final_type, final_value = processed_parts[-1]
    if final_type == 'index':
        idx = final_value
        while len(current) <= idx:
            current.append({})
        current[idx] = value
    else:
        current[final_value] = value


def prepare_hparam_config(original_config_dict, hparam_config_path, completeds=[]):
    import copy
    with open(hparam_config_path, "r") as f:
        hparam_config = yaml.safe_load(f)

    # Extract all hparam paths and their value lists
    hparam_paths = extract_hparam_combinations(hparam_config)
    # Get paths and their value lists
    paths = [path for path, _ in hparam_paths]
    value_lists = [values for _, values in hparam_paths]
    
    # Generate all possible combinations using Cartesian product
    config_dicts = []
    for combination in itertools.product(*value_lists):
        # Start with a deep copy of the original config
        if combination not in completeds:
            config_dict = copy.deepcopy(original_config_dict)
            
            # Set each value in the nested structure
            for path, value in zip(paths, combination):
                set_nested_value(config_dict, path, value)
            
            config_dicts.append(config_dict)
    
    return config_dicts

def run_saturn_hit(cfg_path):
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
        type=int
    )
    parser.add_argument(
        "--oracle_url",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "--reward_type",
        required=False,
        type=str,
        default="geam"
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
        "--jobs_per_gpu",
        type=int,
        required=False,
        default=1,
        help="Number of jobs to run in parallel on each GPU"
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        required=False,
        default=None,
        help="Optional list of GPU device IDs to use (e.g., --devices 0 1 2). If not provided, uses n_gpus or auto-detects available GPUs."
    )
    args = parser.parse_args()

    job_dir = get_job_dir(args.hparam_config is not None, cat=f"saturn-hit")
    print(f"job_dir: {job_dir}")
    
    # Determine which devices to use
    if args.devices is not None:
        # Use specified device IDs
        device_ids = args.devices
        n_gpus = len(device_ids)
        print(f"Using specified devices: {device_ids} ({n_gpus} GPU(s))")
    else:
        # Fall back to auto-detection or n_gpus
        device_ids = None
        if args.direct:
            # Detect all available GPUs when using LocalExecutor
            available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if available_gpus > 0:
                device_ids = list(range(available_gpus))
                n_gpus = available_gpus
                print(f"Detected {n_gpus} available GPU(s) for LocalExecutor: {device_ids}")
            else:
                device_ids = list(range(args.n_gpus))
                n_gpus = args.n_gpus
                print(f"No CUDA devices detected, using n_gpus={n_gpus} from argument: {device_ids}")
        else:
            device_ids = list(range(args.n_gpus))
            n_gpus = args.n_gpus
            print(f"Using n_gpus={n_gpus} from argument: {device_ids}")
    
    if args.direct:
        executor = submitit.LocalExecutor(folder=f"{job_dir}/%j")
        executor.update_parameters(
            timeout_min=10 * 60,
            gpus_per_node=n_gpus,
            nodes=1,
            mem_gb=80,
            cpus_per_task=32
        )
    else:
        executor = submitit.AutoExecutor(folder=f"{job_dir}/%j")
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

    completed_configs = [('hit', False, 64.0), ('hit', False, 128.0)]
    #completed_configs = []
    with open(args.config, "r") as f:
        orig_config_dict = yaml.safe_load(f)
    if args.hparam_config is not None:
        config_dicts = prepare_hparam_config(orig_config_dict, args.hparam_config, completed_configs)
    else:
        config_dicts = [orig_config_dict]

    print(f"{len(config_dicts)} configs prepared to run")
    
    config_paths = []
    for config_dict_idx, config_dict in enumerate(config_dicts):
        model_name = args.config.split("/")[-1].split(".")[0]
        base_log_dir = get_log_dir(
            method=f"saturn",
            model_name=model_name,
            exp_name="exp",
            suffix="-hparam" if args.hparam_config else ""
        )
        os.makedirs(base_log_dir, exist_ok=True)
        # Use the same base log directory for all config_dicts
        log_dir = base_log_dir

        for oracle_name in args.oracle_names:
            oracle_log_dir = os.path.join(log_dir, oracle_name)
            os.makedirs(oracle_log_dir, exist_ok=True)

            for seed in args.seeds:
                seed_log_dir = os.path.join(oracle_log_dir, f"seed-{seed}")
                os.makedirs(seed_log_dir, exist_ok=True)

                # Create a fresh copy of config for each seed
                seed_config_dict = copy.deepcopy(config_dict)
                experiment_id = generate_random_hash(8)
                # Update config with run-specific settings
                seed_config_dict["seed"] = seed
                #seed_config_dict["oracle"]["components"][0]["name"] = args.reward_type
                seed_config_dict["oracle"]["components"][0]["specific_parameters"]["target"] = oracle_name
                if args.oracle_url is not None:
                    seed_config_dict["oracle"]["components"][0]["specific_parameters"]["oracle_url"] = args.oracle_url
                seed_config_dict["logging"]["logging_path"] = os.path.join(seed_log_dir, "results.log")
                seed_config_dict["logging"]["model_checkpoints_dir"] = seed_log_dir
                seed_config_dict["oracle"]["budget"] = args.max_oracle_calls
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
                    job = executor.submit(run_saturn_hit, cfg_path=cfg_path)
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
                job = executor.submit(run_saturn_hit, cfg_path=cfg_path)
                print(job)
                jobs.append(job)    