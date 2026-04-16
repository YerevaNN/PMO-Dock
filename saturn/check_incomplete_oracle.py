#!/usr/bin/env python3
"""
Rerun experiments where oracle_history.csv shows < min_oracle_calls oracle calls spent.
Empties each incomplete seed folder and runs the experiment with the appropriate config.

Usage:
  export PROJECT_ROOT=/home/molopt/Even-More-PMO  # required
  cd $PROJECT_ROOT/Saturn
  python rerun_incomplete_oracle.py --base_results_path /data/molopt/results/saturn/spec/2026-02-05-hparam
"""
import os
import glob
import argparse

if not os.environ.get("PROJECT_ROOT"):
    raise RuntimeError("PROJECT_ROOT environment variable must be set (e.g. export PROJECT_ROOT=/home/molopt/Even-More-PMO)")


MIN_ORACLE_CALLS = 3000


def find_incomplete_runs(base_results_path, min_oracle_calls=MIN_ORACLE_CALLS):
    """
    Find all seed directories where the last row of oracle_history.csv
    has first column (oracle_calls) < min_oracle_calls.
    Returns list of (seed_path, config_path) tuples.
    """
    incomplete = []
    oracle_csv = "oracle_history.csv"
    all_jobs = 0
    for exp_dir in sorted(glob.glob(os.path.join(base_results_path, "exp-*"))):
        for task_name in sorted(os.listdir(exp_dir)):
            task_path = os.path.join(exp_dir, task_name)
            if not os.path.isdir(task_path):
                continue
            for seed_dir in sorted(os.listdir(task_path)):
                if not seed_dir.startswith("seed-"):
                    continue
                seed_path = os.path.join(task_path, seed_dir)
                if not os.path.isdir(seed_path):
                    continue

                csv_path = os.path.join(seed_path, oracle_csv)
                last_val = 0
                if os.path.isfile(csv_path):
                    try:
                        with open(csv_path, "r") as f:
                            lines = [l for l in f.readlines() if l.strip()]
                        if len(lines) >= 2:
                            last_row = lines[-1]
                            first_col = last_row.split(",")[0].strip()
                            last_val = int(first_col) if first_col.isdigit() else 0
                    except Exception:
                        pass
                # No csv, empty csv, or last_val < min: all count as incomplete (including emptied folders from prior reruns)

                if last_val < min_oracle_calls:
                    try:
                        seed_num = int(seed_dir.split("-")[-1])
                    except ValueError:
                        continue
                    config_path = os.path.join(task_path, f"config-{seed_num}.yaml")
                    if os.path.isfile(config_path):
                        incomplete.append((seed_path, config_path, last_val))
                all_jobs += 1
    print(f"All jobs: {all_jobs}")
    return incomplete


def empty_seed_folder(seed_path):
    """Remove generated files from the seed folder (logs, csv, json, ckpt)."""
    removed = []

    # results.log
    log_path = os.path.join(seed_path, "results.log")
    if os.path.isfile(log_path):
        os.remove(log_path)
        removed.append(log_path)

    # *.csv
    for f in glob.glob(os.path.join(seed_path, "*.csv")):
        os.remove(f)
        removed.append(f)

    # *.json
    for f in glob.glob(os.path.join(seed_path, "*.json")):
        os.remove(f)
        removed.append(f)

    # *.ckpt
    for f in glob.glob(os.path.join(seed_path, "*.ckpt")):
        os.remove(f)
        removed.append(f)

    return removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rerun experiments with < min_oracle_calls in oracle_history.csv"
    )
    parser.add_argument("--base_results_path", type=str, required=True)
    parser.add_argument("--min_oracle_calls", type=int, default=MIN_ORACLE_CALLS)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print each incomplete run (seed_path, config_path, last_oracle_calls)",
        )
    args = parser.parse_args()
    incomplete = find_incomplete_runs(args.base_results_path, args.min_oracle_calls)
    print(f"Found {len(incomplete)} incomplete runs")
    if args.list and incomplete:
        for seed_path, config_path, last_val in incomplete:
            print(f"  {seed_path} (last_oracle_calls={last_val}) -> {config_path}")
