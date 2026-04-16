import argparse
import os
import sys
import yaml
import subprocess
import submitit

# Add project root to path for utils imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.experiment_utils import (
    get_job_dir,
    get_log_dir,
    generate_random_hash
)


def run_genmol_lead(cfg_path):
    genmol_lead_command = [
        "python3",
        "scripts/exps/lead/run.py",
        "--config_file",
        cfg_path
    ]

    print(" ".join(genmol_lead_command))
    genmol_lead_process = subprocess.Popen(genmol_lead_command)
    genmol_lead_process.wait()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
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
        "-t",
        "--sim_thresholds",
        nargs="+",
        required=True,
        type=float
    )
    parser.add_argument(
        "-i",
        "--start_mol_indices",
        nargs="+",
        required=True,
        type=int
    )
    parser.add_argument(
        "--tox",
        action="store_true",
        default=False
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
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--hparam_config",
        type=str,
        required=False,
        default=None
    )
    args = parser.parse_args()

    job_dir = get_job_dir(args.hparam_config is not None, cat="genmol-lead")
    print(f"job_dir: {job_dir}")
    

    if args.direct:
        executor = submitit.LocalExecutor(folder=f"{job_dir}/%j")
        n_gpus = args.n_gpus
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
        # name="llama-chem",
        slurm_job_name=os.environ["SLURM_JOB_NAME"],
        slurm_account=os.environ["SLURM_ACCOUNT"],
        timeout_min=2 * 60,
        slurm_array_parallelism=10,
        # slurm_gres=f"gpu:{n_gpus}",
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=80,
        cpus_per_task=20,
        slurm_additional_parameters=slurm_additional_parameters,
    )

    jobs = []
    args_dict = vars(args)
    with open(args.config_file, "r") as f:
        config_dicts = [yaml.safe_load(f)]

    for orig_config_dict in config_dicts:
        oracle_names = args_dict['oracle_names']
        # create log dirs
        model_name = args.config_file.split("/")[-1].split(".")[0]
        log_dir = get_log_dir(
            method="genetic-genmol",
            model_name=model_name,
            exp_name="exp",
            suffix=f"-hparam" if args.hparam_config else ""
        )
        os.makedirs(log_dir, exist_ok=True)
        
        config_log_dirs = []
        config_paths = []
        for oracle_name in oracle_names:
            oracle_log_dir = os.path.join(log_dir, oracle_name)
            os.makedirs(oracle_log_dir, exist_ok=True)

            for sim_thr in args.sim_thresholds:
                sim_thr_log_dir = os.path.join(oracle_log_dir, f"sim_thr-{sim_thr}")
                os.makedirs(sim_thr_log_dir, exist_ok=True)
                for start_mol_idx in args.start_mol_indices:
                    start_mol_idx_log_dir = os.path.join(sim_thr_log_dir, f"start_mol_idx-{start_mol_idx}")
                    os.makedirs(start_mol_idx_log_dir, exist_ok=True)
                    for device_ind, seed in enumerate(args.seeds):
                        seed_log_dir = os.path.join(start_mol_idx_log_dir, f"seed-{seed}")
                        os.makedirs(seed_log_dir, exist_ok=True)

                        # Create a fresh copy of config for each seed
                        config_dict = orig_config_dict.copy()
                        experiment_id = generate_random_hash(8)
                        
                        # Update config with run-specific settings
                        config_dict["seed"] = seed
                        config_dict["oracle_name"] = oracle_name
                        config_dict["log_dir"] = seed_log_dir
                        config_dict["sim_thr"] = sim_thr
                        config_dict["start_mol_idx"] = start_mol_idx
                        config_dict["tox"] = args.tox
                        # For local execution, use cuda:0 for all seeds if n_gpus=1
                        if args.direct and n_gpus == 1:
                            config_dict["device"] = "cuda:0"
                        else:
                            config_dict["device"] = f"cuda:{device_ind % n_gpus}"

                        cfg_file = os.path.join(start_mol_idx_log_dir, f"config-{seed}.yaml")
                        with open(cfg_file, "w") as f:
                            yaml.safe_dump(config_dict, f)
                        config_paths.append(cfg_file)

                if args.direct:
                    for cfg_path in config_paths:
                        # Submit one job that will handle all seeds
                        job = executor.submit(
                            run_genmol_lead,
                            cfg_path=cfg_path
                        )
                        job.result()
                        print(job)
                        jobs.append(job)
                
                else:
                    with executor.batch():
                        for cfg_path in config_paths:
                            job = executor.submit(
                                run_genmol_lead,
                                cfg_path=cfg_path
                            )
                            print(job)
                            jobs.append(job)