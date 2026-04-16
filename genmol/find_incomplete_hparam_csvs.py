#!/usr/bin/env python3
"""
Find incomplete CSV files from hparam search results.
A CSV is incomplete if it has fewer than 3000 unique molecules.
Structure: hparam_folder/exp-*/<task>/seed-*/results.csv
"""

import csv
from pathlib import Path

from benchmark.paths import resolve_from_project_root


HPARAM_DIR = resolve_from_project_root("results", "genetic-genmol", "genmol_hit", "2026-01-15-hparam")
MIN_UNIQUE_MOLECULES = 3000


def count_unique_molecules(csv_path: Path) -> int:
    """Count unique molecules in first column of CSV (excluding header)."""
    molecules = set()
    try:
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    molecules.add(row[0].strip())
    except Exception as e:
        return -1  # Signal error
    return len(molecules)


def main():
    incomplete = []
    errors = []

    for csv_path in sorted(HPARAM_DIR.rglob("results.csv")):
        n_unique = count_unique_molecules(csv_path)
        if n_unique < 0:
            errors.append((str(csv_path), "read error"))
        elif n_unique < MIN_UNIQUE_MOLECULES:
            rel_path = csv_path.relative_to(HPARAM_DIR)
            incomplete.append((str(rel_path), n_unique))

    # Report
    print(f"Hparam dir: {HPARAM_DIR}")
    print(f"Threshold: >= {MIN_UNIQUE_MOLECULES} unique molecules")
    print()

    if incomplete:
        print(f"Found {len(incomplete)} incomplete CSV(s):\n")
        for rel_path, n in incomplete:
            print(f"  {n:5d} unique  {rel_path}")
    else:
        print("No incomplete CSV files found.")

    if errors:
        print(f"\n{len(errors)} error(s) reading files:")
        for p, msg in errors:
            print(f"  {p}: {msg}")


if __name__ == "__main__":
    main()
