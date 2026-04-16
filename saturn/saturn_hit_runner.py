import argparse
import copy
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


def prepare_hparam_config(original_config_dict, hparam_config_path):
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
        config_dict = copy.deepcopy(original_config_dict)
        
        # Set each value in the nested structure
        for path, value in zip(paths, combination):
            set_nested_value(config_dict, path, value)
        
        config_dicts.append(config_dict)
    
    return config_dicts

def run_hit(cfg_path):
    """Run one Saturn job; GPU visibility comes from the scheduler (e.g. SLURM)."""
    cfg_abs = os.path.abspath(cfg_path)
    with open(cfg_abs, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["device"] = "cuda:0"
    temp_cfg = cfg_abs + ".tmp"
    with open(temp_cfg, "w") as f:
        yaml.safe_dump(cfg, f)

    repo = os.environ.get("PROJECT_ROOT", project_root)
    saturn_py = os.path.abspath(os.path.join(repo, "saturn", "saturn.py"))
    saturn_hit_command = [
        sys.executable,
        saturn_py,
        "--config",
        temp_cfg,
    ]

    saturn_hit_process = subprocess.Popen(saturn_hit_command, env=os.environ.copy())
    saturn_hit_process.wait()
    
    # Cleanup
    if os.path.exists(temp_cfg):
        os.remove(temp_cfg)
    
    return saturn_hit_process.returncode

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
        "--reward_type",
        type=str,
        required=False,
        default="geam"
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
        nargs=2,
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

    job_dir = get_job_dir(args.hparam_config is not None, cat=f"saturn-hit")
    logging.info("GPU selection: environment / SLURM; each subprocess uses cuda:0 in config.")

    root_dir = os.path.join(os.environ["PROJECT_ROOT"], "saturn")
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
            method=f"saturn",
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
                seed_config_dict = copy.deepcopy(config_dict)

                # Update config with run-specific settings
                seed_config_dict["seed"] = seed
                
                # Ensure nested structure exists before accessing
                if "oracle" not in seed_config_dict:
                    seed_config_dict["oracle"] = {}
                if "components" not in seed_config_dict["oracle"]:
                    seed_config_dict["oracle"]["components"] = [{}]
                if len(seed_config_dict["oracle"]["components"]) == 0:
                    seed_config_dict["oracle"]["components"] = [{}]
                # Ensure components[0] is a dict, not a string or other type
                if not isinstance(seed_config_dict["oracle"]["components"][0], dict):
                    seed_config_dict["oracle"]["components"][0] = {}
                if "specific_parameters" not in seed_config_dict["oracle"]["components"][0]:
                    seed_config_dict["oracle"]["components"][0]["specific_parameters"] = {}
                # Ensure specific_parameters is a dict
                if not isinstance(seed_config_dict["oracle"]["components"][0]["specific_parameters"], dict):
                    seed_config_dict["oracle"]["components"][0]["specific_parameters"] = {}
                
                seed_config_dict["oracle"]["components"][0]["specific_parameters"]["target"] = oracle
                
                if "logging" not in seed_config_dict:
                    seed_config_dict["logging"] = {}
                seed_config_dict["logging"]["logging_path"] = os.path.join(seed_log_dir, "results.log")
                seed_config_dict["logging"]["model_checkpoints_dir"] = seed_log_dir
                
                seed_config_dict["oracle"]["budget"] = args.max_oracle_calls
                if args.oracle_url is not None:
                    seed_config_dict["oracle"]["components"][0]["specific_parameters"]["oracle_url"] = args.oracle_url
                if not args.hparam_config:
                    # Update reward type if not using hparam config
                    seed_config_dict["oracle"]["components"][0]["name"] = args.reward_type
                
                seed_config_dict["device"] = "cuda:0"

                cfg_file = os.path.join(oracle_log_dir, f"config-{seed}.yaml")
                with open(cfg_file, "w") as f:
                    yaml.safe_dump(seed_config_dict, f)
                all_cfg_paths.append(cfg_file)

    total_workers = args.max_workers
    logging.info(f"Running {len(all_cfg_paths)} jobs with up to {total_workers} parallel workers")
    
    run_hits(cfg_paths=all_cfg_paths, max_workers=total_workers)
    logging.info(f"Completed {len(all_cfg_paths)} jobs")
