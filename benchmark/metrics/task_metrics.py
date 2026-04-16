from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HitRange:
    low: float
    high: float


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Present: {list(df.columns)}")


def _as_numpy(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def metric_hit_rate(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    max_calls: int | None = None,
) -> float:
    """
    Generic hit-rate metric: percent of rows with score==1.
    Intended for hit/dock/spec tasks after you have per-row binary hits.
    """
    _require_cols(df, [score_col])
    s = df[score_col].to_numpy()
    hits = float(np.nansum(s == 1))
    denom = float(max_calls) if max_calls is not None else float(len(s))
    if denom <= 0:
        return 0.0
    return 100.0 * hits / denom


def metric_lead_topk_mean(
    df: pd.DataFrame,
    *,
    docking_col: str = "docking_score",
    k: int = 1,
) -> float:
    """
    Lead metric: mean docking score of the top-k molecules (by docking_score).
    Assumes larger docking_score is better (matches your reward-space convention).
    """
    _require_cols(df, [docking_col])
    if df.empty:
        return float("nan")
    vals = pd.to_numeric(df[docking_col], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    k = max(1, int(k))
    top = np.sort(vals)[-k:]
    return float(np.mean(top))


def metric_spec_margin_topk_mean(
    df: pd.DataFrame,
    *,
    docking_col: str = "docking_score",
    antitarget_col: str = "antitarget_docking_score",
    k: int = 10,
) -> float:
    """
    Specificity metric: mean of top-k margins (docking - antitarget_docking).
    """
    _require_cols(df, [docking_col, antitarget_col])
    if df.empty:
        return float("nan")
    dock = pd.to_numeric(df[docking_col], errors="coerce").to_numpy(dtype=float)
    anti = pd.to_numeric(df[antitarget_col], errors="coerce").to_numpy(dtype=float)
    margin = dock - anti
    margin = margin[~np.isnan(margin)]
    if margin.size == 0:
        return float("nan")
    k = max(1, int(k))
    top = np.sort(margin)[-k:]
    return float(np.mean(top))


def compute_binary_hits(
    df: pd.DataFrame,
    *,
    prop_cols: Sequence[str],
    hit_ranges: Mapping[str, HitRange],
    out_col: str = "hit",
) -> pd.DataFrame:
    """
    Given property columns and per-property hit ranges, compute a per-row binary hit.
    A row is a hit if all listed properties fall inside their ranges.
    """
    _require_cols(df, list(prop_cols))
    work = df.copy()
    ok = np.ones(len(work), dtype=bool)
    for col in prop_cols:
        r = hit_ranges[col]
        v = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        ok &= (v >= float(r.low)) & (v <= float(r.high))
    work[out_col] = ok.astype(int)
    return work

