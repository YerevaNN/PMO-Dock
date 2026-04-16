from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from benchmark.paths import resolve_from_project_root


@lru_cache(maxsize=1)
def load_actives_csv(path: str | None = None) -> List[dict]:
    """
    Load `benchmark/actives.csv` as a list of rows (dicts).
    """
    p = Path(path) if path is not None else resolve_from_project_root("benchmark", "actives.csv")
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


@lru_cache(maxsize=1)
def actives_smiles_by_target(path: str | None = None) -> Dict[str, List[str]]:
    """
    Return target -> ordered smiles list (row order).
    """
    rows = load_actives_csv(path)
    out: Dict[str, List[str]] = {}
    for r in rows:
        t = (r.get("target") or "").strip()
        s = (r.get("smiles") or "").strip()
        if not t or not s:
            continue
        out.setdefault(t, []).append(s)
    return out

