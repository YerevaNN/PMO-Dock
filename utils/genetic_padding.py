"""Padding helpers for top-n lead / spec metrics when fewer than n hits exist."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def mean_top_n_padded(values: pd.Series, n: int, pad_value: float | None) -> float:
    """Mean of the top ``n`` values (descending). If fewer than ``n`` values exist, repeat
    ``pad_value`` to fill up to ``n`` before averaging.

    If there are zero values and ``pad_value`` is None, returns NaN.
    """
    s = values.dropna().astype(float).sort_values(ascending=False)
    got = list(s.head(n).values)
    if len(got) == 0 and pad_value is None:
        return float("nan")
    while len(got) < n:
        if pad_value is None:
            return float("nan")
        got.append(float(pad_value))
    return float(np.mean(got[:n]))


def seed_smiles_from_similar_column(csv_path: Path) -> str | None:
    """Parse SIMILAR.<smiles> column name from genetic oracle CSV header (lead tasks)."""
    for sep in (";", ","):
        try:
            hdr = pd.read_csv(csv_path, sep=sep, nrows=0)
        except Exception:
            continue
        for c in hdr.columns:
            cs = str(c).strip()
            if cs.startswith("SIMILAR."):
                return cs[len("SIMILAR.") :]
    return None


def first_docking_for_smiles(df: pd.DataFrame, smiles: str, invalid: float = 99.9) -> float | None:
    """First-row docking_score for exact molecule match (seed ligand in log)."""
    if smiles is None or not smiles:
        return None
    m = df[df["molecule"].astype(str) == smiles]
    if len(m) == 0:
        return None
    d = float(m.iloc[0]["docking_score"])
    if np.isnan(d) or d == invalid:
        return None
    return d


def load_padding_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_actives_oracle_table(path: Path | None) -> pd.DataFrame | None:
    """Load ``benchmark/actives.csv``-style table (``target``, ``DS``, ...)."""
    if path is None or not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if not {"target", "DS"}.issubset(df.columns):
        return None
    return df


def dock_pad_from_actives_table(
    actives: pd.DataFrame,
    target: str,
    ligand_index: int,
) -> float | None:
    """Docking score from actives row ``ligand_index`` among rows with given ``target`` (order as in file)."""
    sub = actives.loc[actives["target"].astype(str).str.lower() == str(target).lower()]
    if ligand_index < 0 or ligand_index >= len(sub):
        return None
    v = pd.to_numeric(sub.iloc[ligand_index]["DS"], errors="coerce")
    if pd.isna(v):
        return None
    return float(v)


def lead_dock_pad_smiles_only(mols_csv: Path, df_norm: pd.DataFrame) -> float | None:
    """First docking row matching seed SMILES from ``SIMILAR.*`` header (no JSON)."""
    smi = seed_smiles_from_similar_column(mols_csv)
    if smi:
        v = first_docking_for_smiles(df_norm, smi)
        if v is not None:
            return v
    return None


def resolve_lead_seed_dock(
    task_folder: str,
    target_protein: str,
    ligand_index: int,
    mols_csv: Path,
    df_norm: pd.DataFrame,
    padding: dict[str, Any],
    actives: pd.DataFrame | None,
) -> float | None:
    """Seed dock for padding: JSON ``dock_pad``, else ``actives`` ``DS``, else SMILES match in ``df_norm``."""
    o = padding.get(task_folder) or {}
    if "dock_pad" in o:
        return float(o["dock_pad"])
    if actives is not None:
        v = dock_pad_from_actives_table(actives, target_protein, ligand_index)
        if v is not None:
            return v
    return lead_dock_pad_smiles_only(mols_csv, df_norm)


def lead_dock_pad(
    task_folder: str,
    mols_csv: Path,
    df_norm: pd.DataFrame,
    padding: dict[str, Any],
) -> float | None:
    """Docking score used to pad top-n lead mean: JSON override, else first row for seed SMILES."""
    o = padding.get(task_folder) or {}
    if "dock_pad" in o:
        return float(o["dock_pad"])
    smi = seed_smiles_from_similar_column(mols_csv)
    if smi:
        v = first_docking_for_smiles(df_norm, smi)
        if v is not None:
            return v
    return None


def spec_margin_pad(task_folder: str, padding: dict[str, Any]) -> float | None:
    o = padding.get(task_folder) or {}
    if "margin_pad" in o:
        return float(o["margin_pad"])
    return None
