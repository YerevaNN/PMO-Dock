import argparse
import os
import sys
import yaml
import subprocess
import itertools
import logging
import time
import copy
import pandas as pd
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

def run_lead(cfg_path):
    """Run one Saturn lead job; GPU visibility from scheduler / environment."""
    cfg_abs = os.path.abspath(cfg_path)
    with open(cfg_abs, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["device"] = "cuda:0"
    temp_cfg = cfg_abs + ".tmp"
    with open(temp_cfg, "w") as f:
        yaml.safe_dump(cfg, f)

    repo = os.environ.get("PROJECT_ROOT", project_root)
    saturn_py_path = os.path.abspath(os.path.join(repo, "saturn", "saturn.py"))
    saturn_dir = os.path.abspath(os.path.join(repo, "saturn"))

    saturn_command = [
        sys.executable,
        saturn_py_path,
        "--config",
        temp_cfg,
    ]

    saturn_process = subprocess.Popen(saturn_command, env=os.environ.copy(), cwd=saturn_dir)
    saturn_process.wait()
    
    # Cleanup
    if os.path.exists(temp_cfg):
        os.remove(temp_cfg)
    
    return saturn_process.returncode

def run_leads(cfg_paths, max_workers=None):
    """Run multiple lead experiments in parallel in a single job"""
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
    
    logging.info(f"Starting {len(cfg_paths)} lead experiments with {max_workers} parallel workers")
    
    def run_single_lead(cfg_path):
        """Helper function to run a single lead experiment"""
        try:
            return_code = run_lead(cfg_path)
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
    
    # Run leads in parallel using ThreadPoolExecutor with limited workers
    delay = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cfg = {}
        for i, cfg_path in enumerate(cfg_paths):
            if i > 0:
                time.sleep(delay)
            future = executor.submit(run_single_lead, cfg_path)
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
        "--config",
        required=True,
        type=str
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        type=int
    )
    parser.add_argument(
        "--oracle_names",
        nargs="+",
        required=True,
        type=str
    )
    parser.add_argument(
        "--max_oracle_calls",
        required=True,
        type=int
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
        default=None
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
        "--max_workers",
        type=int,
        required=False,
        default=15,
        help="Maximum number of parallel workers for running lead experiments (default: 15)"
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    job_dir = get_job_dir(args.hparam_config is not None, cat=f"saturn-lead")
    logging.info("GPU selection: environment / SLURM; each subprocess uses cuda:0 in config.")

    with open(args.config, "r") as f:
        orig_config_dict = yaml.safe_load(f)
    if args.hparam_config is not None:
        config_dicts = prepare_hparam_config(orig_config_dict, args.hparam_config)
    else:
        config_dicts = [copy.deepcopy(orig_config_dict)]
    
    lead_smiles_df = pd.read_csv(
        os.path.join(os.environ["PROJECT_ROOT"], "saturn", "lead", "actives.csv")
    )

    all_cfg_paths = []

    for config_dict in config_dicts:
        # create log dirs
        model_name = args.config.split("/")[-1].split(".")[0]
        log_dir = get_log_dir(
            method=f"saturn",
            model_name=model_name,
            exp_name="exp",
            suffix="-hparam" if args.hparam_config else ""
        )
        os.makedirs(log_dir, exist_ok=True)
        
        for start_mol_idx in args.start_mol_idx:
            start_idx_dir = os.path.join(log_dir, f"start_mol_idx-{start_mol_idx}")
            os.makedirs(start_idx_dir, exist_ok=True)

            for oracle_name in args.oracle_names:
                start_smiles = lead_smiles_df[lead_smiles_df["target"] == oracle_name]['smiles'].iloc[start_mol_idx]
                oracle_log_dir = os.path.join(start_idx_dir, oracle_name)
                os.makedirs(oracle_log_dir, exist_ok=True)

                for seed in args.seeds:
                    seed_log_dir = os.path.join(oracle_log_dir, f"seed-{seed}")
                    os.makedirs(seed_log_dir, exist_ok=True)

                    # Create a deep copy to avoid modifying the original
                    seed_config_dict = copy.deepcopy(config_dict)
                    
                        # Ensure oracle structure exists
                    if "oracle" not in seed_config_dict:
                        seed_config_dict["oracle"] = {}
                    # Ensure oracle is a dict (not a string or other type)
                    if not isinstance(seed_config_dict["oracle"], dict):
                        seed_config_dict["oracle"] = {}
                    if "components" not in seed_config_dict["oracle"]:
                        seed_config_dict["oracle"]["components"] = [{}]
                    # Ensure components is a list (not a string or other type)
                    if not isinstance(seed_config_dict["oracle"]["components"], list):
                        seed_config_dict["oracle"]["components"] = [{}]
                    if len(seed_config_dict["oracle"]["components"]) == 0:
                        seed_config_dict["oracle"]["components"] = [{}]
                    
                    # Ensure components[0] is a dict (not a string or other type)
                    if not isinstance(seed_config_dict["oracle"]["components"][0], dict):
                        seed_config_dict["oracle"]["components"][0] = {}
                    if "specific_parameters" not in seed_config_dict["oracle"]["components"][0]:
                        seed_config_dict["oracle"]["components"][0]["specific_parameters"] = {}
                    if not isinstance(seed_config_dict["oracle"]["components"][0]["specific_parameters"], dict):
                        seed_config_dict["oracle"]["components"][0]["specific_parameters"] = {}
                    
                    # Update config with run-specific settings
                    seed_config_dict["seed"] = seed
                    seed_config_dict["oracle"]["components"][0]["specific_parameters"]["target"] = oracle_name
                    if args.oracle_url is not None:
                        seed_config_dict["oracle"]["components"][0]["specific_parameters"]["oracle_url"] = args.oracle_url
                    # Ensure logging structure exists
                    if "logging" not in seed_config_dict:
                        seed_config_dict["logging"] = {}
                    if not isinstance(seed_config_dict["logging"], dict):
                        seed_config_dict["logging"] = {}
                    seed_config_dict["logging"]["logging_path"] = os.path.join(seed_log_dir, "results.log")
                    seed_config_dict["logging"]["model_checkpoints_dir"] = seed_log_dir
                    seed_config_dict["oracle"]["budget"] = args.max_oracle_calls
                    
                    # Ensure goal_directed_generation structure exists
                    if "goal_directed_generation" not in seed_config_dict:
                        seed_config_dict["goal_directed_generation"] = {}
                    if not isinstance(seed_config_dict["goal_directed_generation"], dict):
                        seed_config_dict["goal_directed_generation"] = {}
                    if "experience_replay" not in seed_config_dict["goal_directed_generation"]:
                        seed_config_dict["goal_directed_generation"]["experience_replay"] = {}
                    if not isinstance(seed_config_dict["goal_directed_generation"]["experience_replay"], dict):
                        seed_config_dict["goal_directed_generation"]["experience_replay"] = {}
                    if "hallucinated_memory" not in seed_config_dict["goal_directed_generation"]:
                        seed_config_dict["goal_directed_generation"]["hallucinated_memory"] = {}
                    if not isinstance(seed_config_dict["goal_directed_generation"]["hallucinated_memory"], dict):
                        seed_config_dict["goal_directed_generation"]["hallucinated_memory"] = {}
                    if "reinforcement_learning" not in seed_config_dict["goal_directed_generation"]:
                        seed_config_dict["goal_directed_generation"]["reinforcement_learning"] = {}
                    if not isinstance(seed_config_dict["goal_directed_generation"]["reinforcement_learning"], dict):
                        seed_config_dict["goal_directed_generation"]["reinforcement_learning"] = {}
                    
                    seed_config_dict["goal_directed_generation"]["experience_replay"]["lead_smiles_in_buffer"] = start_smiles
                    seed_config_dict["oracle"]["components"][0]["lead_smiles"] = start_smiles
                    seed_config_dict["goal_directed_generation"]["experience_replay"]["fix_lead_smiles_in_buffer"] = args.fix_lead_smiles_in_buffer
                    if args.execute_hallucinated_memory is not None:
                        seed_config_dict["goal_directed_generation"]["hallucinated_memory"]["execute_hallucinated_memory"] = args.execute_hallucinated_memory
                    if args.sigma is not None:
                        seed_config_dict["goal_directed_generation"]["reinforcement_learning"]["sigma"] = args.sigma
                    
                    seed_config_dict["device"] = "cuda:0"

                    cfg_file = os.path.join(oracle_log_dir, f"config-{seed}.yaml")
                    # Normalize the path to ensure consistency
                    cfg_file = os.path.abspath(cfg_file)
                    
                    with open(cfg_file, "w") as f:
                        yaml.safe_dump(seed_config_dict, f)
                    all_cfg_paths.append(cfg_file)

    total_workers = args.max_workers
    logging.info(f"Running {len(all_cfg_paths)} lead experiments with up to {total_workers} parallel workers")
    
    run_leads(cfg_paths=all_cfg_paths, max_workers=total_workers)
    logging.info(f"Completed {len(all_cfg_paths)} lead experiments")
