#!/usr/bin/env python3
"""
Local multi-GPU "job imitation" runner for multi_objective genetic_gfn hit-task grid search.

Use when Slurm is unavailable but you have a multi-GPU node (e.g., 8x H100).

It enumerates the 3x3x3 grid:
  targets (default): parp1, fa7, 5ht1b
  kl_coefficient values and rank_coefficient values from genetic_gfn/hparams_tune.yaml

For each grid cell it runs ONE process per seed (default 10 seeds) using task=simple:
  python run.py genetic_gfn --objectives qed,sa,<target> --alpha_vector 1,1,1
    --max_oracle_calls 3000 --task simple --seed <seed> --freq_log 100 --wandb disabled

Concurrency:
  Runs up to N processes in parallel. GPU visibility is inherited from the environment (do not set
  CUDA_VISIBLE_DEVICES here). Default limits are conservative (one concurrent run); raise
  --max_parallel only if you intend to share one GPU (e.g. MPS) or have set devices externally.

Output:
  Writes under a single output root directory:
    - results/CSVs:  <out_dir>/genetic_gfn/results/hit_task_grid/   (flat output dir; filenames encode settings)
    - logs:          <out_dir>/genetic_gfn/local_jobs/multi_objective_hit_grid/<run_name>.gpu<G>.log
  Also uses a per-run DOCKING_TMP_DIR under the run’s output folder to avoid collisions.
"""

import argparse
import os
import subprocess
import sys
import time
import datetime
import traceback
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple, Optional

import yaml

from benchmark.paths import resolve_from_project_root


@dataclass(frozen=True)
class Cell:
    target: str
    kl: float
    rank: float


DEFAULT_OUT_DIR = str(resolve_from_project_root("results"))


def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def dirs() -> Tuple[str, str]:
    # <REPO_ROOT>/multi_objective/run_hit_task_grid_local.py
    script_path = abspath(__file__)
    multi_obj_dir = os.path.dirname(script_path)
    repo_root = os.path.dirname(multi_obj_dir)
    return repo_root, multi_obj_dir


def parse_hparams(hparam_yaml: str) -> Tuple[List[float], List[float]]:
    with open(hparam_yaml, "r") as f:
        hp = yaml.safe_load(f) or {}
    params = (hp.get("parameters") or {}) if isinstance(hp, dict) else {}
    kl_vals = (((params.get("kl_coefficient") or {}).get("values")) or [])
    rank_vals = (((params.get("rank_coefficient") or {}).get("values")) or [])
    if not kl_vals or not rank_vals:
        raise ValueError(
            f"Invalid hparam config: {hparam_yaml}. "
            "Expected parameters.kl_coefficient.values and parameters.rank_coefficient.values."
        )
    return [float(x) for x in kl_vals], [float(x) for x in rank_vals]


def parse_seeds(hparam_yaml: str) -> List[int]:
    """
    Optional: read seeds from the same hparam yaml (top-level key `seeds`).
    Falls back to [0..9] if not present.
    """
    with open(hparam_yaml, "r") as f:
        hp = yaml.safe_load(f) or {}
    seeds = []
    if isinstance(hp, dict):
        seeds = hp.get("seeds") or []
    if not seeds:
        return list(range(10))
    return [int(s) for s in seeds]


def write_config(base_cfg: str, out_cfg: str, *, kl: float, rank: float) -> None:
    with open(base_cfg, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["kl_coefficient"] = float(kl)
    cfg["rank_coefficient"] = float(rank)
    os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
    with open(out_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def build_cells(targets: List[str], kl_vals: List[float], rank_vals: List[float]) -> List[Cell]:
    return [Cell(t, float(kl), float(rank)) for t, kl, rank in product(targets, kl_vals, rank_vals)]


def launch(
    *,
    cell: Cell,
    seed: int,
    slot_id: int,
    multi_obj_dir: str,
    out_dir: str,
    run_date_dir: str,
    max_oracle_calls: int,
    freq_log: int,
) -> subprocess.Popen:
    results_root = os.path.join(out_dir, "genetic_gfn", "results")
    logs_root = os.path.join(out_dir, "genetic_gfn", "local_jobs", "multi_objective_hit_grid")
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    run_name = f"{cell.target}_hit_task_kl{cell.kl}_rank{cell.rank}_seed{seed}"
    # Flat output directory: filenames (run_name + seed suffixes) encode settings.
    run_out = os.path.join(results_root, "hit_task_grid", run_date_dir)
    os.makedirs(run_out, exist_ok=True)

    base_cfg = os.path.join(multi_obj_dir, "genetic_gfn", "hparams_default.yaml")
    cfg_out = os.path.join(run_out, f"hparams_{run_name}.yaml")
    write_config(base_cfg, cfg_out, kl=cell.kl, rank=cell.rank)

    cmd = [
        # Unbuffered stdout/stderr so log files update in real time
        sys.executable,
        "-u",
        "run.py",
        "genetic_gfn",
        "--objectives",
        f"qed,sa,{cell.target}",
        "--alpha_vector",
        "1,1,1",
        "--max_oracle_calls",
        str(max_oracle_calls),
        "--task",
        "simple",
        "--seed",
        str(seed),
        "--freq_log",
        str(freq_log),
        "--wandb",
        "disabled",
        "--run_name",
        run_name,
        "--output_dir",
        run_out,
        "--config_default",
        cfg_out,
    ]

    log_path = os.path.join(logs_root, run_date_dir, f"{run_name}.slot{slot_id}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Line-buffer the log file to make progress visible immediately.
    log_f = open(log_path, "w", buffering=1)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Do NOT set DOCKING_TMP_DIR here; multi_objective/optimizer.py will set a per-run temp dir
    # based on task_label and delete it after the run finishes.

    log_f.write(f"CWD: {multi_obj_dir}\n")
    log_f.write(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n")
    log_f.write("CMD: " + " ".join(cmd) + "\n\n")
    log_f.flush()

    p = subprocess.Popen(cmd, cwd=multi_obj_dir, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    p._log_f = log_f  # type: ignore[attr-defined]
    p._log_path = log_path  # type: ignore[attr-defined]
    p._run_name = run_name  # type: ignore[attr-defined]
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["parp1", "fa7", "5ht1b"])
    ap.add_argument("--hparam_config", default="genetic_gfn/hparams_tune.yaml")
    ap.add_argument("--max_oracle_calls", type=int, default=3000)
    ap.add_argument("--freq_log", type=int, default=100)
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Root output directory for results + logs")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional explicit seed list (overrides hparam yaml seeds)")

    ap.add_argument(
        "--gpu_ids",
        nargs="*",
        type=int,
        default=None,
        help="Concurrency slot indices for scheduling/logs (not CUDA device ids).",
    )
    ap.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of concurrent slots if --gpu_ids not provided (default 1: safe single-GPU).",
    )
    ap.add_argument(
        "--max_parallel",
        type=int,
        default=1,
        help="Cap on concurrent runs (default 1 unless sharing one GPU intentionally).",
    )
    ap.add_argument("--poll_sec", type=float, default=2.0)
    args = ap.parse_args()

    out_dir = abspath(args.out_dir)

    _, multi_obj_dir = dirs()

    # Date folder to group results/logs by day (and time to avoid collisions if you run multiple times/day)
    now = datetime.datetime.now()
    run_date_dir = now.strftime("%Y-%m-%d")
    run_time_tag = now.strftime("%H%M%S")
    run_date_dir = f"{run_date_dir}_{run_time_tag}"

    hparam_path = args.hparam_config
    if not os.path.isabs(hparam_path):
        hparam_path = os.path.join(multi_obj_dir, hparam_path)
    kl_vals, rank_vals = parse_hparams(hparam_path)
    cells = build_cells(args.targets, kl_vals, rank_vals)
    seeds = args.seeds if args.seeds else parse_seeds(hparam_path)

    gpu_ids = args.gpu_ids if args.gpu_ids else list(range(args.num_gpus))
    max_parallel = min(args.max_parallel, len(gpu_ids))

    print("OUT_DIR:", out_dir)
    print("MULTI_OBJECTIVE_DIR:", multi_obj_dir)
    print("NUM_CELLS:", len(cells))
    print("SEEDS:", seeds)
    print("CONCURRENCY_SLOTS:", gpu_ids)
    print("MAX_PARALLEL:", max_parallel)
    print("RUN_DATE_DIR:", run_date_dir)

    # Scheduler logs + summary
    results_root = os.path.join(out_dir, "genetic_gfn", "results", "hit_task_grid", run_date_dir)
    logs_root = os.path.join(out_dir, "genetic_gfn", "local_jobs", "multi_objective_hit_grid", run_date_dir)
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)
    scheduler_log_path = os.path.join(logs_root, "scheduler.log")
    summary_csv_path = os.path.join(logs_root, "runs_summary.csv")

    def slog(msg: str) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(scheduler_log_path, "a") as f:
            f.write(line + "\n")

    def append_summary(
        *,
        run_name: str,
        cell: Cell,
        seed: int,
        gpu_id: int,
        pid: Optional[int],
        start_ts: float,
        end_ts: float,
        return_code: Optional[int],
        signal: Optional[int],
        log_path: str,
    ) -> None:
        # Write one CSV row and flush immediately (so we keep a record even if the scheduler dies).
        with open(summary_csv_path, "a") as f:
            f.write(
                f"{run_name},{cell.target},{cell.kl},{cell.rank},{seed},{gpu_id},{pid if pid is not None else ''},"
                f"{start_ts:.3f},{end_ts:.3f},{return_code if return_code is not None else ''},"
                f"{signal if signal is not None else ''},{log_path}\n"
            )
            f.flush()

    if not os.path.exists(summary_csv_path):
        with open(summary_csv_path, "w") as f:
            f.write("run_name,target,kl,rank,seed,gpu_id,pid,start_time,end_time,return_code,signal,log_path\n")

    # Expand to per-seed jobs
    queue: List[Tuple[Cell, int]] = [(cell, s) for cell in cells for s in seeds]
    running: List[Tuple[subprocess.Popen, int, Cell, int, float]] = []  # (proc, gpu, cell, seed, start_ts)

    def poll_finished():
        nonlocal running
        try:
            still = []
            for p, gid, cell, seed, start_ts in running:
                rc = p.poll()
                if rc is None:
                    still.append((p, gid, cell, seed, start_ts))
                    continue

                try:
                    p._log_f.flush()  # type: ignore[attr-defined]
                    p._log_f.close()  # type: ignore[attr-defined]
                except Exception:
                    pass

                end_ts = time.time()
                sig: Optional[int] = None
                if isinstance(rc, int) and rc < 0:
                    sig = -rc

                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                log_path = getattr(p, "_log_path", "")
                run_name = getattr(p, "_run_name", f"{cell.target}_seed{seed}")
                slog(f"[done] {status} slot={gid} pid={p.pid} run={run_name}")

                append_summary(
                    run_name=run_name,
                    cell=cell,
                    seed=seed,
                    gpu_id=gid,
                    pid=p.pid,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    return_code=rc,
                    signal=sig,
                    log_path=log_path,
                )

                # If there's no useful stdout, common causes are SIGKILL (OOM killer) or external preemption.
                if rc != 0 and sig in (9, 15):
                    slog(
                        f"[hint] run={run_name} ended by signal {sig}. If the log has no traceback, check OOM killer / system logs:\n"
                        f"       - dmesg -T | tail -200\n"
                        f"       - journalctl -k --since '1 hour ago' | tail -200\n"
                        f"       - nvidia-smi (GPU resets/ECC)\n"
                    )

            running = still
        except Exception as e:
            # Never let scheduler die due to a logging/inspection issue.
            slog("[scheduler-error] poll_finished exception:\n" + traceback.format_exc())

    while queue or running:
        poll_finished()

        while queue and len(running) < max_parallel:
            used = {gid for _, gid, _, _, _ in running}
            free = [g for g in gpu_ids if g not in used]
            if not free:
                break
            gid = free[0]
            cell, seed = queue.pop(0)
            # Record start time before launching so failures still get accounted for.
            start_ts = time.time()
            run_name = f"{cell.target}_hit_task_kl{cell.kl}_rank{cell.rank}_seed{seed}"
            try:
                p = launch(
                    cell=cell,
                    seed=seed,
                    slot_id=gid,
                    multi_obj_dir=multi_obj_dir,
                    out_dir=out_dir,
                    run_date_dir=run_date_dir,
                    max_oracle_calls=args.max_oracle_calls,
                    freq_log=args.freq_log,
                )
                slog(f"[start] slot={gid} pid={p.pid} run={getattr(p, '_run_name', run_name)} cell={cell} seed={seed}")
                running.append((p, gid, cell, seed, start_ts))
            except Exception:
                end_ts = time.time()
                slog(f"[launch-failed] slot={gid} run={run_name} cell={cell} seed={seed}\n" + traceback.format_exc())
                append_summary(
                    run_name=run_name,
                    cell=cell,
                    seed=seed,
                    gpu_id=gid,
                    pid=None,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    return_code=None,
                    signal=None,
                    log_path="",
                )

        time.sleep(args.poll_sec)

    slog("All runs finished.")


if __name__ == "__main__":
    main()

