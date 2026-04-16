#!/usr/bin/env python3
"""Submitit launcher for per-hparam array jobs.

Desired layout:
- 1 SLURM job per fixed hparam config (search_range i i+1)
- inside that job, run all (target, seed) pairs in parallel via genetic_runner --max_workers
- with 2 targets x 3 seeds and max_workers=6, all six runs can execute concurrently
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

try:
    import submitit
except Exception as e:  # pragma: no cover
    submitit = None
    _submitit_import_error = e


_GC_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_GC_ROOT)
sys.path.insert(0, _REPO_ROOT)

from utils.experiment_utils import get_job_dir  # noqa: E402


def _require_submitit() -> None:
    if submitit is None:  # pragma: no cover
        raise ImportError(
            "submitit is not available in this environment. "
            "Install it (pip install submitit) or run on a cluster image that includes it. "
            f"Original error: {_submitit_import_error}"
        )


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _to_nested_dict(orig_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same flattening convention as `genetic_runner.py`:
      {"genetic": {"pool_size": [10, 20]}} -> {"genetic.pool_size": [10, 20]}
    """
    nested: Dict[str, Any] = {}
    for k, v in orig_dict.items():
        if isinstance(v, dict):
            for nk, nv in v.items():
                nested[f"{k}.{nk}"] = nv
        else:
            nested[k] = v
    return nested


def count_hparam_combinations(hparam_config_path: str) -> int:
    hp = _read_yaml(hparam_config_path)
    flat = _to_nested_dict(hp)
    keys = list(flat.keys())
    if not keys:
        return 1

    values: List[List[Any]] = []
    for k in keys:
        v = flat[k]
        if isinstance(v, list):
            values.append(v)
        else:
            # genetic_runner assumes lists; be forgiving here
            values.append([v])

    # Small (64) in your use-case; explicit product keeps behavior clear.
    return sum(1 for _ in itertools.product(*values))


def _pick_vina_url(vina_urls: Sequence[str], hparam_idx: int) -> Optional[str]:
    if not vina_urls:
        return None
    return str(vina_urls[hparam_idx % len(vina_urls)])


@dataclass(frozen=True)
class HparamJob:
    project_root: str
    config_file: str
    hparam_config: str
    oracle_names: List[str]
    seeds: List[int]
    max_oracle_calls: int
    reward_type: str
    max_workers: int
    hparam_idx: int
    vina_url: Optional[str]
    extra_env: Dict[str, str]

    def __call__(self) -> int:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PROJECT_ROOT"] = self.project_root
        env["PYTHONPATH"] = f"{self.project_root}:{env.get('PYTHONPATH', '')}"
        env.update(self.extra_env)

        cmd = [
            sys.executable,
            "-u",
            os.path.join(self.project_root, "genetic_chemalactica", "genetic_runner.py"),
            "--config_file",
            self.config_file,
            "--hparam_config",
            self.hparam_config,
            "--oracle_name",
            *self.oracle_names,
            "--seeds",
            *[str(s) for s in self.seeds],
            "--max_oracle_calls",
            str(self.max_oracle_calls),
            "--reward_type",
            str(self.reward_type),
            "--max_workers",
            str(self.max_workers),
            "--search_range",
            str(self.hparam_idx),
            str(self.hparam_idx + 1),
        ]

        if self.vina_url:
            cmd += ["--vina_url", str(self.vina_url)]

        print("HPARAM_JOB_SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID", ""))
        print("HPARAM_JOB_ORACLE_NAMES:", self.oracle_names)
        print("HPARAM_JOB_SEEDS:", self.seeds)
        print("HPARAM_JOB_MAX_WORKERS:", self.max_workers)
        print("HPARAM_JOB_SEARCH_RANGE:", (self.hparam_idx, self.hparam_idx + 1))
        print("HPARAM_JOB_VINA_URL:", self.vina_url or "")
        print("CMD:", " ".join(cmd))
        sys.stdout.flush()

        # Run from genetic_chemalactica/ for relative imports & file expectations.
        workdir = os.path.join(self.project_root, "genetic_chemalactica")
        p = subprocess.Popen(cmd, cwd=workdir, env=env)
        p.wait()
        return int(p.returncode or 0)


def main() -> None:
    _require_submitit()

    parser = argparse.ArgumentParser()

    # Functional args mirroring genetic_runner.py
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--hparam_config", required=True, type=str)
    parser.add_argument("--oracle_name", nargs="+", required=True, type=str)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--max_oracle_calls", required=True, type=int)
    parser.add_argument("--reward_type", default="hit", type=str)
    parser.add_argument("--max_workers", default=5, type=int)
    parser.add_argument("--hparam_start", default=0, type=int, help="Inclusive start hparam index.")
    parser.add_argument("--hparam_end", default=None, type=int, help="Exclusive end hparam index.")

    # Oracle assignment
    parser.add_argument(
        "--vina_urls",
        nargs="*",
        default=[],
        type=str,
        help="List of remote oracle service base URLs. Hparam jobs are assigned round-robin.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned jobs (ranges + vina_url) without submitting.",
    )

    # SLURM / submitit args
    parser.add_argument("--partition", required=True, type=str)
    parser.add_argument("--job_name", default="chem_genetic_hparam", type=str)
    parser.add_argument("--account", default=None, type=str)
    parser.add_argument("--qos", default=None, type=str)
    parser.add_argument("--timeout_min", default=48 * 60, type=int)
    parser.add_argument("--mem_gb", default=64, type=int)
    parser.add_argument("--cpus_per_task", default=4, type=int)
    parser.add_argument("--gpus_per_node", default=1, type=int)

    parser.add_argument(
        "--hparam_array_parallelism",
        default=64,
        type=int,
        help="Max concurrent hparam arrays across submitit batches.",
    )

    # Output folders
    parser.add_argument(
        "--out_dir",
        default=os.environ.get("OUT_DIR", ""),
        type=str,
        help="Sets OUT_DIR for get_job_dir() and downstream logging.",
    )
    parser.add_argument(
        "--project_root",
        default=os.environ.get("PROJECT_ROOT", _REPO_ROOT),
        type=str,
        help="Repository root. Defaults to $PROJECT_ROOT or autodetected.",
    )

    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        # Default to <PROJECT_ROOT>/results for public reproducibility.
        from benchmark.paths import resolve_from_project_root

        out_dir = str(resolve_from_project_root("results"))

    os.environ["OUT_DIR"] = out_dir
    os.environ["PROJECT_ROOT"] = project_root

    # Normalize paths so submitted jobs don't depend on caller CWD.
    config_file = args.config_file
    if not os.path.isabs(config_file):
        config_file = os.path.join(project_root, config_file)
    hparam_config = args.hparam_config
    if not os.path.isabs(hparam_config):
        hparam_config = os.path.join(project_root, hparam_config)

    total = count_hparam_combinations(hparam_config)
    hparam_start = max(0, int(args.hparam_start))
    hparam_end = total if args.hparam_end is None else min(total, int(args.hparam_end))
    if hparam_end <= hparam_start:
        raise ValueError(
            f"Invalid hparam slice: start={hparam_start}, end={hparam_end}, total={total}"
        )
    runs_per_hparam = len(args.oracle_name) * len(args.seeds)

    print("PROJECT_ROOT:", project_root)
    print("OUT_DIR:", out_dir)
    print("CONFIG_FILE:", config_file)
    print("HPARAM_CONFIG:", hparam_config)
    print("TOTAL_HPARAM_CONFIGS:", total)
    print("HPARAM_RANGE:", (hparam_start, hparam_end))
    print("NUM_HPARAM_SUBMITTED:", hparam_end - hparam_start)
    print("RUNS_PER_HPARAM (targets x seeds):", runs_per_hparam)
    print("MAX_WORKERS_PER_HPARAM_JOB:", int(args.max_workers))
    print("VINA_URLS_COUNT:", len(args.vina_urls))
    for i in range(hparam_start, hparam_end):
        print(
            f"  hparam[{i}]: search_range=({i}, {i + 1}) "
            f"vina_url={_pick_vina_url(args.vina_urls, i - hparam_start) or ''}"
        )

    if args.dry_run:
        return

    job_dir = get_job_dir(is_hparam_search=True, cat="chem-hparam-submitit")
    os.makedirs(job_dir, exist_ok=True)
    submitted_jobs: List[Tuple[int, Any]] = []
    extra_env = {
        "OUT_DIR": out_dir,
        "PROJECT_ROOT": project_root,
    }
    # 1 hparam -> 1 SLURM job; it runs 6 runs internally via max_workers.
    for local_idx, hparam_idx in enumerate(range(hparam_start, hparam_end)):
        vina_url = _pick_vina_url(args.vina_urls, local_idx)
        hparam_dir = os.path.join(job_dir, f"hparam_{hparam_idx:03d}")
        # One folder per hparam and one subfolder per submitted slurm job id.
        submitit_root = os.path.join(hparam_dir, "%j")
        os.makedirs(hparam_dir, exist_ok=True)

        executor = submitit.AutoExecutor(folder=submitit_root)
        slurm_additional_parameters = {"partition": args.partition}
        params: Dict[str, Any] = dict(
            slurm_job_name=f"{args.job_name}_{hparam_idx:03d}",
            timeout_min=int(args.timeout_min),
            gpus_per_node=int(args.gpus_per_node),
            nodes=1,
            mem_gb=int(args.mem_gb),
            cpus_per_task=int(args.cpus_per_task),
            slurm_additional_parameters=slurm_additional_parameters,
        )
        if args.account:
            params["slurm_account"] = args.account
        if args.qos:
            params["slurm_qos"] = args.qos
        executor.update_parameters(**params)

        job = executor.submit(
            HparamJob(
                project_root=project_root,
                config_file=config_file,
                hparam_config=hparam_config,
                oracle_names=list(args.oracle_name),
                seeds=list(args.seeds),
                max_oracle_calls=int(args.max_oracle_calls),
                reward_type=str(args.reward_type),
                max_workers=int(args.max_workers),
                hparam_idx=int(hparam_idx),
                vina_url=vina_url,
                extra_env=extra_env,
            )
        )
        submitted_jobs.append((hparam_idx, job))

    print("")
    print("Submitted hparam jobs (1 job per hparam):")
    for hparam_idx, job in submitted_jobs:
        print(f"  hparam_{hparam_idx:03d}: job_id={job.job_id} stdout={job.paths.stdout}")
    print("")
    print("Log layout:")
    print(f"  {job_dir}/hparam_XXX/<jobid>/")
    print("Each job internally runs all target/seed runs in parallel via --max_workers.")


if __name__ == "__main__":
    main()

