#!/usr/bin/env python3
"""Export per-run lead top-1 scores (padded) for a scale folder.

Writes a CSV inside the given ``scale_root`` with columns:
  target_protein, sim_tier (04/06), ligand_index (0/1/2), random_seed (seed-#), score

Score definition:
- **Hits** use only **QED** and **SA** bounds from the lead sim tier (``lead.sim_04`` / ``lead.sim_06``
  in ``utils/tasks.py``), not similarity or docking cutoffs.
- If hits exist: score = max(docking_score) over hits (cumulative over whole run).
- If no hits: pad with seed docking in order: JSON ``dock_pad``, then ``DS`` from
  ``benchmark/actives.csv`` (row ``ligand_index`` among rows for that ``target``), then
  first docking row for the reference SMILES from the ``SIMILAR.<smiles>`` column name.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.compute_metrics import filter_df  # noqa: E402
from utils.genetic_experiment_metrics import (  # noqa: E402
    INVALID_DOCK,
    normalize_columns,
    property_columns_from_mols_header,
    read_molecules_csv,
)
from utils.genetic_padding import (  # noqa: E402
    load_padding_json,
    read_actives_oracle_table,
    resolve_lead_seed_dock,
)
from utils.tasks import lead_qed_sa_hit_constraints  # noqa: E402


LEAD_FOLDER_RE = re.compile(r"^lead\.(?P<protein>[^_]+)_(?P<sim>04|06)_(?P<idx>[012])$")


def parse_lead_folder(name: str) -> tuple[str, str, str] | None:
    m = LEAD_FOLDER_RE.match(name)
    if not m:
        return None
    return m.group("protein"), m.group("sim"), m.group("idx")


def find_mols_csv(seed_dir: Path) -> Path | None:
    for name in ("0_mols.csv", "mols.csv"):
        p = seed_dir / name
        if p.is_file():
            return p
    return None


def compute_top1_padded(df_norm: pd.DataFrame, constraints: dict, *, pad: float | None) -> float:
    valid = df_norm["docking_score"].notna() & (df_norm["docking_score"] != INVALID_DOCK)
    base = df_norm.loc[valid].drop_duplicates(subset=["molecule"], keep="first")
    hits = filter_df(base, constraints)
    if len(hits):
        return float(hits["docking_score"].max())
    return float(pad) if pad is not None else float("nan")


def export_scale_root(
    scale_root: Path,
    out_csv: Path,
    *,
    sim_only: str | None,
    padding: dict[str, Any],
    actives: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    for task_dir in sorted(p for p in scale_root.iterdir() if p.is_dir() and p.name.startswith("lead.")):
        parsed = parse_lead_folder(task_dir.name)
        if not parsed:
            continue
        protein, sim, idx = parsed
        if sim_only is not None and sim != sim_only:
            continue
        constraints = lead_qed_sa_hit_constraints(sim)

        for seed_dir in sorted(p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("seed-")):
            mols = find_mols_csv(seed_dir)
            if mols is None:
                continue
            raw = read_molecules_csv(mols)
            comp = property_columns_from_mols_header(mols)
            df_norm = normalize_columns(raw, comp)
            pad = resolve_lead_seed_dock(
                task_dir.name,
                protein,
                int(idx),
                mols,
                df_norm,
                padding,
                actives,
            )
            score = compute_top1_padded(df_norm, constraints, pad=pad)
            rows.append(
                {
                    "target_protein": protein,
                    "sim_tier": sim,
                    "ligand_index": idx,
                    "random_seed": seed_dir.name,
                    "score": score,
                    "seed_dock_pad": pad,
                    "task_folder": task_dir.name,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scale_root", type=Path)
    ap.add_argument("--sim", choices=["04", "06"], default=None, help="Optionally restrict to sim tier")
    ap.add_argument(
        "--out-name",
        default=None,
        help="Output CSV filename (default: lead_top1_scores[_simXX].csv)",
    )
    ap.add_argument(
        "--padding-json",
        type=Path,
        default=None,
        help="Optional JSON with per-task dock_pad overrides (same as genetic metrics scripts)",
    )
    ap.add_argument(
        "--actives-csv",
        type=Path,
        default=_REPO / "benchmark/actives.csv",
        help="Actives table (target, DS, …); ligand_index picks row among that target. Missing file → skip.",
    )
    args = ap.parse_args()
    scale_root = args.scale_root.resolve()
    if not scale_root.is_dir():
        raise SystemExit(f"Not a dir: {scale_root}")
    name = args.out_name
    if name is None:
        name = f"lead_top1_scores{('_sim'+args.sim) if args.sim else ''}.csv"
    out_csv = scale_root / name
    pad_json = load_padding_json(args.padding_json.resolve() if args.padding_json else None)
    actives_path = args.actives_csv.resolve() if args.actives_csv else None
    actives_df = read_actives_oracle_table(actives_path)
    df = export_scale_root(
        scale_root,
        out_csv,
        sim_only=args.sim,
        padding=pad_json,
        actives=actives_df,
    )
    print(f"Wrote {len(df)} rows -> {out_csv}")


if __name__ == "__main__":
    main()

