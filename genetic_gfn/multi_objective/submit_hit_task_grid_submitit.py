#!/usr/bin/env python3
"""
Array-of-packers submitit runner for GeneticGFN hit tasks.

Goal:
- You have MANY runs (e.g., 54) but each uses ~0.5 GPU.
- Slurm GPUs are exclusive (no MIG / no shared GPUs).
- So we request 4 GPUs per Slurm array task ("packer"), then inside that job
  we run 2 processes per GPU (8 concurrent runs), assigning GPUs round-robin.

What you get:
- Slurm-level visibility: multiple array tasks (packers), not one mega job.
- Per-run logs: seed-*/logs/<run_name>.stdout.log and .stderr.log
- Per-run status markers: seed-*/status/STARTED, DONE, FAILED.txt
"""

import argparse
import itertools
import math
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

try:
    import submitit
except Exception as e:  # pragma: no cover
    submitit = None
    _submitit_import_error = e


# ------------------------
# repo path setup
# This file expected at: <REPO_ROOT>/GeneticGFN/multi_objective/submit_hit_runner_packers_submitit.py
# ------------------------
repo_root = os.environ.get("PROJECT_ROOT")
if not repo_root:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

from utils.experiment_utils import get_log_dir  # noqa: E402


# ------------------------
# Selectivity configuration: target -> list of anti-target docking receptors.
# ------------------------
TARGET_TO_ANTI_TARGETS = {
    "6nzp": ["7uyt", "5ut5"],
}


def _require_submitit():
    if submitit is None:  # pragma: no cover
        raise ImportError(
            "submitit is not installed. Install it (pip install submitit) or run on a cluster image that includes it. "
            f"Original error: {_submitit_import_error}"
        )


def _safe_name(s: str) -> str:
    return str(s).replace(os.sep, "_").replace(" ", "_")


def prepare_hparam_config(original_config_dict: Dict[str, Any], hparam_config_path: str) -> List[Dict[str, Any]]:
    """
    Generic sweep YAML: mapping key -> scalar or list.
      e.g.
        kl_coefficient: [0.1, 0.3]
        rank_coefficient: [0.5, 1.0]
        population_size: 128
    """
    with open(hparam_config_path, "r") as f:
        hp = yaml.safe_load(f) or {}

    keys = list(hp.keys())
    values = [hp[k] if isinstance(hp[k], list) else [hp[k]] for k in keys]

    out = []
    for combo in itertools.product(*values):
        cfg = dict(original_config_dict)
        for k, v in zip(keys, combo):
            cfg[k] = v
        out.append(cfg)
    return out


def _count_visible_gpus_fallback(expected: int) -> int:
    """
    Slurm usually sets CUDA_VISIBLE_DEVICES to something like "3,7,2,5".
    Inside the job, the process sees those as cuda:0..cuda:(n-1).
    We'll count how many were allocated to decide round-robin range.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cvd = (cvd or "").strip()
    if cvd == "":
        return max(1, int(expected))
    # Some systems may set it to "NoDevFiles" or similar; handle conservatively.
    if cvd.lower() in {"nodevfiles", "none"}:
        return max(1, int(expected))
    return len([x for x in cvd.split(",") if x.strip() != ""])


def _write_status(run_dir: str, name: str, text: Optional[str] = None) -> None:
    status_dir = os.path.join(run_dir, "status")
    os.makedirs(status_dir, exist_ok=True)
    p = os.path.join(status_dir, name)
    if text is None:
        with open(p, "w") as f:
            f.write("")
    else:
        with open(p, "w") as f:
            f.write(str(text))


def _build_command_from_run_config(cfg: Dict[str, Any], project_root: str) -> List[str]:
    cmd = [
        "python3",
        "-u",
        f"{project_root}/GeneticGFN/multi_objective/run.py",
        "genetic_gfn",
        "--task", "simple",
        "--seed", str(cfg["seed"]),
        "--objectives", str(cfg["objectives"]),
        "--alpha_vector", str(cfg.get("alpha_vector", "1,1,1")),
        "--max_oracle_calls", str(cfg["max_oracle_calls"]),
        "--freq_log", str(cfg.get("freq_log", 100)),
        "--output_dir", str(cfg["output_dir"]),
        "--run_name", str(cfg["run_name"]),
        "--config_default", str(cfg["config_default_path"]),
    ]

    anti_target = (cfg.get("anti_target") or "").strip()
    if anti_target:
        cmd += ["--anti_target", anti_target]

    # Pass through CPU pool size to run.py (joblib parallelism), if requested.
    if cfg.get("n_jobs") is not None:
        try:
            n_jobs = int(cfg["n_jobs"])
            if n_jobs != -1:
                cmd += ["--n_jobs", str(n_jobs)]
        except Exception:
            pass

    if cfg.get("oracle_url"):
        cmd += ["--oracle_url", str(cfg["oracle_url"])]

    return cmd


def _run_one_cfg(cfg_path: str, *, project_root: str, local_gpu_index: int, multi_obj_dir: str) -> Tuple[str, int]:
    """
    Run exactly one run_config.yaml on exactly one GPU (relative index within allocation),
    writing per-run logs + status markers. Returns (cfg_path, returncode).
    """
    run_dir = os.path.dirname(os.path.abspath(cfg_path))

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Per-run logs
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    run_name_safe = _safe_name(cfg.get("run_name") or f"run_{os.path.basename(run_dir)}")

    stdout_path = os.path.join(logs_dir, f"{run_name_safe}.stdout.log")
    stderr_path = os.path.join(logs_dir, f"{run_name_safe}.stderr.log")

    # Status
    _write_status(run_dir, "STARTED", f"gpu={local_gpu_index}\nstart={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    cmd = _build_command_from_run_config(cfg, project_root)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Critical: pin this subprocess to ONE of the GPUs allocated to the packer job.
    # local_gpu_index is 0..(num_visible_gpus-1) relative to the packer allocation.
    parent_cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if parent_cvd:
        ids = [x.strip() for x in parent_cvd.split(",") if x.strip() != ""]
        if 0 <= local_gpu_index < len(ids):
            env["CUDA_VISIBLE_DEVICES"] = ids[local_gpu_index]
        else:
            # fallback: don't override
            env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        # fallback (non-slurm / no restriction)
        env["CUDA_VISIBLE_DEVICES"] = str(local_gpu_index)

    with open(stdout_path, "w", buffering=1) as out_f, open(stderr_path, "w", buffering=1) as err_f:
        out_f.write(f"CWD: {multi_obj_dir}\n")
        out_f.write(f"PACKER_SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID','')}\n")
        out_f.write(f"PARENT_CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES','')}\n")
        out_f.write(f"CHILD_CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES','')}\n")
        out_f.write("CMD: " + " ".join(cmd) + "\n\n")
        out_f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=multi_obj_dir,
            env=env,
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
        p.wait()
        rc = int(p.returncode)

    if rc == 0:
        _write_status(run_dir, "DONE", f"end={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        _write_status(run_dir, "FAILED.txt", f"exit_code={rc}\nend={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    return cfg_path, rc


@dataclass(frozen=True)
class PackerJob:
    """
    One Slurm array task: runs a chunk of run_config.yaml files.

    Inside the allocated GPUs, we run:
      concurrent = jobs_per_gpu * num_visible_gpus
    and assign GPUs round-robin.
    """
    cfg_paths_chunk: List[str]
    project_root: str
    expected_gpus: int
    jobs_per_gpu: int
    start_delay_sec: float

    def __call__(self) -> Dict[str, Any]:
        multi_obj_dir = os.path.join(self.project_root, "GeneticGFN", "multi_objective")

        num_gpus = _count_visible_gpus_fallback(self.expected_gpus)
        num_gpus = max(1, num_gpus)
        jobs_per_gpu = max(1, int(self.jobs_per_gpu))
        max_concurrent = num_gpus * jobs_per_gpu

        print("PACKER_SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID", ""))
        print("PARENT_CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        print("VISIBLE_GPUS_COUNT:", num_gpus)
        print("JOBS_PER_GPU:", jobs_per_gpu)
        print("MAX_CONCURRENT:", max_concurrent)
        print("NUM_RUNS_IN_CHUNK:", len(self.cfg_paths_chunk))

        results = []
        start = time.time()

        # We’ll launch up to max_concurrent processes; chunk size is typically <= max_concurrent
        with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
            futs = []
            for i, cfg_path in enumerate(self.cfg_paths_chunk):
                local_gpu = i % num_gpus
                if self.start_delay_sec and i > 0:
                    time.sleep(float(self.start_delay_sec))
                futs.append(
                    ex.submit(
                        _run_one_cfg,
                        cfg_path,
                        project_root=self.project_root,
                        local_gpu_index=local_gpu,
                        multi_obj_dir=multi_obj_dir,
                    )
                )

            for fut in as_completed(futs):
                cfg_path, rc = fut.result()
                results.append((cfg_path, rc))

        ok = sum(1 for _, rc in results if rc == 0)
        bad = len(results) - ok
        elapsed_min = (time.time() - start) / 60.0

        summary = {
            "ok": ok,
            "failed": bad,
            "elapsed_min": elapsed_min,
            "num_gpus_visible": num_gpus,
            "jobs_per_gpu": jobs_per_gpu,
            "max_concurrent": max_concurrent,
            "results": results,  # list of (cfg_path, rc)
        }
        print("PACKER_SUMMARY:", summary)
        return summary


def build_all_run_configs(args) -> Tuple[List[str], str]:
    """
    Builds run_config.yaml files and returns:
      - list of cfg paths
      - submitit_root (where submitit logs will go)
    """
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        raise RuntimeError("PROJECT_ROOT env var is required")

    targets = args.targets if args.targets else args.oracle_name
    if not targets:
        raise ValueError("--targets is required (or deprecated --oracle_name)")

    multi_obj_root = os.path.join(project_root, "genetic_gfn", "multi_objective")

    base_cfg_path = args.config_file
    if not os.path.isabs(base_cfg_path):
        base_cfg_path = os.path.join(multi_obj_root, base_cfg_path)

    with open(base_cfg_path, "r") as f:
        orig_config_dict = yaml.safe_load(f) or {}

    if args.hparam_config:
        hparam_path = args.hparam_config
        if not os.path.isabs(hparam_path):
            hparam_path = os.path.join(multi_obj_root, hparam_path)
        config_dicts = prepare_hparam_config(orig_config_dict, hparam_path)
    else:
        config_dicts = [orig_config_dict]

    # Experiment root + submitit logs root
    model_name = os.path.basename(args.config_file).split(".")[0]
    exp_root = get_log_dir(
        method="genetic_gfn",
        model_name=model_name,
        exp_name="exp",
        suffix="-hparam" if args.hparam_config else "",
    )
    run_date_dir = time.strftime("%Y-%m-%d_%H%M%S")
    submitit_root = os.path.join(exp_root, "slurm_jobs", "submitit", "hit_packers", run_date_dir)
    os.makedirs(submitit_root, exist_ok=True)

    all_cfg_paths: List[str] = []

    for config_dict in config_dicts:
        os.makedirs(exp_root, exist_ok=True)

        for target in targets:
            anti_targets = TARGET_TO_ANTI_TARGETS.get(str(target), [])
            anti_iter = anti_targets if anti_targets else [""]

            for anti_target in anti_iter:
                target_dir = os.path.join(exp_root, str(target))
                if anti_target:
                    target_dir = os.path.join(target_dir, f"anti-{anti_target}")
                os.makedirs(target_dir, exist_ok=True)

                for seed in args.seeds:
                    seed_log_dir = os.path.join(target_dir, f"seed-{seed}")
                    os.makedirs(seed_log_dir, exist_ok=True)

                    # (1) GeneticGFN hyperparam config_default.yaml for this run
                    config_default_dict = dict(config_dict)
                    if args.oracle_url:
                        config_default_dict["oracle_url"] = args.oracle_url
                    config_default_path = os.path.join(seed_log_dir, "config_default.yaml")
                    with open(config_default_path, "w") as f:
                        yaml.safe_dump(config_default_dict, f, sort_keys=False)

                    # (2) runner meta-config for this run
                    run_name = f"{target}_hit_task_seed{seed}"
                    if anti_target:
                        run_name = f"{run_name}_anti-{anti_target}"

                    cfg_out = {
                        "seed": int(seed),
                        "target": str(target),
                        "anti_target": str(anti_target) if anti_target else "",
                        "objectives": f"{args.objectives_prefix},{target}",
                        "alpha_vector": str(args.alpha_vector),
                        "oracle_url": str(args.oracle_url) if args.oracle_url else "",
                        "max_oracle_calls": int(args.max_oracle_calls),
                        "freq_log": int(args.freq_log),
                        "output_dir": seed_log_dir,
                        "run_name": run_name,
                        "config_default_path": config_default_path,
                        "n_jobs": int(args.n_jobs) if args.n_jobs is not None else -1,
                        # Important: each subprocess will see exactly one GPU and use cuda:0.
                        "device": "cuda:0",
                    }

                    cfg_file = os.path.join(seed_log_dir, "run_config.yaml")
                    with open(cfg_file, "w") as f:
                        yaml.safe_dump(cfg_out, f, sort_keys=False)

                    all_cfg_paths.append(cfg_file)

    return all_cfg_paths, submitit_root


def chunk_list(xs: List[str], chunk_size: int) -> List[List[str]]:
    return [xs[i : i + chunk_size] for i in range(0, len(xs), chunk_size)]


def main():
    _require_submitit()

    parser = argparse.ArgumentParser()

    # --- functional args (same spirit as your runner) ---
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--targets", nargs="+", required=True, type=str)
    parser.add_argument("--oracle_name", nargs="+", required=False, type=str, help="DEPRECATED: use --targets")
    parser.add_argument("--oracle_url", required=False, type=str)
    parser.add_argument("--max_oracle_calls", required=True, type=int)
    parser.add_argument("--n_jobs", required=False, default=-1, type=int)
    parser.add_argument("--alpha_vector", required=False, default="1,1,1", type=str)
    parser.add_argument("--objectives_prefix", required=False, default="qed,sa", type=str)
    parser.add_argument("--freq_log", required=False, default=100, type=int)
    parser.add_argument("--hparam_config", type=str, required=False, default=None)

    # --- packing knobs ---
    parser.add_argument("--jobs_per_gpu", type=int, default=2, help="How many concurrent runs per allocated GPU.")
    parser.add_argument("--start_delay_sec", type=float, default=2.0, help="Delay between launching subprocesses.")

    # --- slurm/submitit args ---
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument("--job_name", type=str, default="genetic_gfn_hit_packers")
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--qos", type=str, default=None)

    parser.add_argument("--gpus_per_packer", type=int, default=4, help="GPUs requested by EACH packer job.")
    parser.add_argument("--cpus_per_task", type=int, default=48)
    parser.add_argument("--mem_gb", type=int, default=128)
    parser.add_argument("--timeout_min", type=int, default=48 * 60)

    parser.add_argument(
        "--slurm_array_parallelism",
        type=int,
        default=1,
        help="How many packer tasks run concurrently. Keep 1 if each packer uses all GPUs you can request.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        default=False,
        help="Run locally via submitit.LocalExecutor (debug).",
    )

    args = parser.parse_args()

    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        raise RuntimeError("PROJECT_ROOT env var is required")

    all_cfg_paths, submitit_root = build_all_run_configs(args)

    # Each packer can run (gpus_per_packer * jobs_per_gpu) runs concurrently.
    pack_capacity = max(1, int(args.gpus_per_packer)) * max(1, int(args.jobs_per_gpu))
    chunks = chunk_list(all_cfg_paths, pack_capacity)

    print("PROJECT_ROOT:", project_root)
    print("SUBMITIT_LOG_ROOT:", submitit_root)
    print("TOTAL_RUNS:", len(all_cfg_paths))
    print("GPUS_PER_PACKER:", args.gpus_per_packer)
    print("JOBS_PER_GPU:", args.jobs_per_gpu)
    print("PACK_CAPACITY:", pack_capacity)
    print("NUM_PACKERS (array tasks):", len(chunks))
    print("ARRAY_PARALLELISM:", args.slurm_array_parallelism)
    print("NOTE: each packer allocates GPUs; keep array_parallelism low enough to fit cluster limits.")

    if args.direct:
        executor = submitit.LocalExecutor(folder=os.path.join(submitit_root, "%j"))
        executor.update_parameters(
            timeout_min=args.timeout_min,
            gpus_per_node=args.gpus_per_packer,
            nodes=1,
            mem_gb=args.mem_gb,
            cpus_per_task=args.cpus_per_task,
        )
    else:
        executor = submitit.AutoExecutor(folder=os.path.join(submitit_root, "%j"))
        params = dict(
            slurm_job_name=args.job_name,
            timeout_min=args.timeout_min,
            slurm_array_parallelism=args.slurm_array_parallelism,
            gpus_per_node=args.gpus_per_packer,
            nodes=1,
            mem_gb=args.mem_gb,
            cpus_per_task=args.cpus_per_task,
            slurm_additional_parameters={"partition": args.partition},
        )
        if args.account:
            params["slurm_account"] = args.account
        if args.qos:
            params["slurm_qos"] = args.qos

        executor.update_parameters(**params)

    submitted = []
    with executor.batch():
        for chunk in chunks:
            job = executor.submit(
                PackerJob(
                    cfg_paths_chunk=chunk,
                    project_root=project_root,
                    expected_gpus=int(args.gpus_per_packer),
                    jobs_per_gpu=int(args.jobs_per_gpu),
                    start_delay_sec=float(args.start_delay_sec),
                )
            )
            submitted.append(job)

    # After batch(), job ids are available
    for j in submitted:
        print("submitted packer job_id:", j.job_id)

    print(f"Submitted {len(submitted)} packer tasks for {len(all_cfg_paths)} runs.")
    print("Per-run logs/status are under each seed-*/logs and seed-*/status.")


if __name__ == "__main__":
    main()
