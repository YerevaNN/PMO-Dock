#!/usr/bin/env python3
"""
Check for GenMol experiments where results.csv has < min_unique_molecules unique molecules,
or where results.csv is missing (empty seed-* folders). Prints the count of incomplete
runs (no files are modified).

Under base_results_path there must be exp-* folders. Any nesting under exp-* is supported
(e.g. exp-0/protein/threshold/1/ or exp-0/task/seed-0/). Run dirs are discovered via
config: any directory named seed-N or N whose parent contains config-{N}.yaml is a run;
missing or undersized results.csv makes it incomplete.

Usage:
  export PROJECT_ROOT=/home/molopt/Even-More-PMO  # required
  cd $PROJECT_ROOT/genmol
  python check_incomplete_genmol.py --base_results_path /data/molopt/results/genetic-genmol/genmol_lead/hparam_search
"""
import csv
import glob
import os
import argparse

if not os.environ.get("PROJECT_ROOT"):
    raise RuntimeError(
        "PROJECT_ROOT environment variable must be set (e.g. export PROJECT_ROOT=/home/molopt/Even-More-PMO)"
    )

MIN_UNIQUE_MOLECULES = 3000


def count_unique_molecules(csv_path):
    """Count unique molecules in first column of CSV (excluding header)."""
    molecules = set()
    try:
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    molecules.add(row[0].strip())
    except Exception:
        return -1
    return len(molecules)


def _parse_run_index(dirname):
    """From a run directory name (e.g. seed-0, 1, 2) return the index for config-{N}.yaml."""
    if dirname.startswith("seed-"):
        try:
            return int(dirname.split("-")[-1])
        except ValueError:
            return None
    if dirname.isdigit():
        return int(dirname)
    return None


def _find_run_dirs_by_config(root_dir):
    """
    Recursively yield (run_path, config_path) for every run dir under root_dir.
    A run dir is a direct child of a directory that contains config-{N}.yaml,
    where the child name is seed-N or N (so empty seed-* folders are included).
    """
    try:
        entries = os.listdir(root_dir)
    except OSError:
        return
    for name in sorted(entries):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        idx = _parse_run_index(name)
        if idx is not None:
            config_path = os.path.join(root_dir, f"config-{idx}.yaml")
            if os.path.isfile(config_path):
                yield path, config_path
        yield from _find_run_dirs_by_config(path)


def find_incomplete_runs(base_results_path, min_unique=MIN_UNIQUE_MOLECULES):
    """
    Find all run directories (any nesting under exp-*) where results.csv has
    < min_unique unique molecules, or results.csv is missing (empty seed-*).
    Returns list of (run_path, config_path, n_unique).
    """
    incomplete = []
    results_csv = "results.csv"
    all_jobs = 0
    exp_dirs = sorted(glob.glob(os.path.join(base_results_path, "exp-*")))
    if not exp_dirs:
        print("All jobs: 0 (no exp-* folders found)")
        return incomplete

    for exp_dir in exp_dirs:
        if not os.path.isdir(exp_dir):
            continue
        for run_path, config_path in _find_run_dirs_by_config(exp_dir):
            csv_path = os.path.join(run_path, results_csv)
            if not os.path.isfile(csv_path):
                n_unique = 0
            else:
                n_unique = count_unique_molecules(csv_path)
            all_jobs += 1
            if n_unique < min_unique:
                incomplete.append((run_path, config_path, n_unique))

    print(f"All jobs: {all_jobs}")
    return incomplete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for GenMol experiments with < min_unique_molecules in results.csv"
    )
    parser.add_argument("--base_results_path", type=str, required=True)
    parser.add_argument("--min_unique_molecules", type=int, default=MIN_UNIQUE_MOLECULES)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print each incomplete run (seed_path, n_unique)",
    )
    args = parser.parse_args()

    incomplete = find_incomplete_runs(
        args.base_results_path, min_unique=args.min_unique_molecules
    )
    print(f"Found {len(incomplete)} incomplete runs")

    if args.list and incomplete:
        for seed_path, config_path, n_unique in incomplete:
            print(f"  {seed_path} (n_unique={n_unique})")
