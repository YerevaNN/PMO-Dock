from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from benchmark.paths import resolve_from_project_root


def actives_csv_path(path: str | Path | None = None) -> Path:
    """Canonical lead-seed table: ``benchmark/actives.csv``."""
    if path is not None:
        return Path(path)
    return resolve_from_project_root("benchmark", "actives.csv")


@lru_cache(maxsize=8)
def load_actives_csv(path: str | None = None) -> List[dict]:
    """
    Load `benchmark/actives.csv` as a list of rows (dicts).
    """
    p = actives_csv_path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


@lru_cache(maxsize=8)
def actives_rows_by_target(path: str | None = None) -> Dict[str, List[dict]]:
    """Return target -> ordered row list (CSV row order)."""
    out: Dict[str, List[dict]] = {}
    for r in load_actives_csv(path):
        t = (r.get("target") or "").strip()
        if not t:
            continue
        out.setdefault(t, []).append(r)
    return out


@lru_cache(maxsize=8)
def actives_smiles_by_target(path: str | None = None) -> Dict[str, List[str]]:
    """
    Return target -> ordered smiles list (row order).
    """
    out: Dict[str, List[str]] = {}
    for t, rows in actives_rows_by_target(path).items():
        smiles = [(r.get("smiles") or "").strip() for r in rows]
        out[t] = [s for s in smiles if s]
    return out


def lead_seed_row(target: str, idx: int, path: str | None = None) -> dict:
    """Return the actives.csv row for `target` at seed index `idx`."""
    by_target = actives_rows_by_target(path)
    if target not in by_target:
        raise KeyError(f"Unknown lead target {target!r}. Known: {sorted(by_target)}")
    rows = by_target[target]
    i = int(idx)
    if i < 0 or i >= len(rows):
        raise IndexError(f"Lead seed index {i} out of range for {target} (n={len(rows)})")
    return rows[i]


def lead_seed_smiles(target: str, idx: int, path: str | None = None) -> str:
    """Return lead-task seed SMILES for `target` at index `idx` (from actives.csv)."""
    smiles = (lead_seed_row(target, idx, path).get("smiles") or "").strip()
    if not smiles:
        raise ValueError(f"Empty SMILES for lead seed {target}[{idx}]")
    return smiles


def lead_seed_ds(target: str, idx: int, path: str | None = None) -> float:
    """Return the recorded docking score (DS) for the lead seed ligand."""
    return float(lead_seed_row(target, idx, path)["DS"])

