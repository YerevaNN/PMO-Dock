"""
Scorer module for genetic_gfn multi-objective optimization.

Docking objectives use one HTTP call per batch via ``DockingOracleClient``.
Antitarget multi-seed docking (7uyt, 5ut5, 7uyw, 4l00, 5khw) is handled inside
``benchmark.docking_oracle.docking.DockingOracle.predict`` on the service.
"""

from __future__ import annotations

import os

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tdc import Oracle as TDCOracle

from benchmark.docking_oracle.docking import TARGET_BOX
from benchmark.docking_oracle.docking_vina_client import DockingOracleClient

_DOCKING_HTTP_URL = ""
_LEAD_SMILES = ""
_ANTI_TARGET = ""

# Metadata for optimizer imports; docking keys mirror benchmark receptors.
TARGET_CONFIGS = {
    target: {
        "box_center": list(box["center"]),
        "box_size": list(box["size"]),
        "receptor_file": f"{target}.pdbqt",
    }
    for target, box in TARGET_BOX.items()
}

DOCKING_WINDOWS = {}

_qed_oracle = None
_sa_oracle = None


def set_docking_service_url(url: str) -> None:
    """Configure docking oracle base URL (e.g. http://127.0.0.1:5050)."""
    global _DOCKING_HTTP_URL
    _DOCKING_HTTP_URL = (url or "").strip()


def set_lead_smiles(smiles: str) -> None:
    """Lead SMILES for the ``similarity`` objective (Tanimoto)."""
    global _LEAD_SMILES
    _LEAD_SMILES = (smiles or "").strip()


def set_anti_target(name: str) -> None:
    """Record anti-target receptor for selectivity runs (optimizer applies reward logic)."""
    global _ANTI_TARGET
    _ANTI_TARGET = (name or "").strip().lower()


def _http_docking_base() -> str:
    url = _DOCKING_HTTP_URL or os.environ.get("DOCKING_VINA_URL") or os.environ.get("ORACLE_SERVICE_URL") or ""
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def _get_qed_oracle():
    global _qed_oracle
    if _qed_oracle is None:
        _qed_oracle = TDCOracle(name="QED")
    return _qed_oracle


def _get_sa_oracle():
    global _sa_oracle
    if _sa_oracle is None:
        _sa_oracle = TDCOracle(name="SA")
    return _sa_oracle


def normalize_docking_score(raw_score, dock_min=-20.0, dock_max=-10.0):
    if raw_score < -100 or raw_score > 0:
        return 0.0
    normalized = (raw_score - dock_min) / (dock_max - dock_min)
    return max(0.0, min(1.0, normalized))


def normalize_sa_score(sa_score, sa_min=1.0, sa_max=5.0):
    if sa_score < 1.0 or sa_score > 10.0:
        return 0.0
    if sa_score <= sa_max:
        normalized = 1.0 - (sa_score - sa_min) / (sa_max - sa_min)
    else:
        normalized = 0.0
    return max(0.0, min(1.0, normalized))


def _smiles_list_from_mols(mols):
    smiles_list = []
    for mol in mols:
        if isinstance(mol, str):
            smiles_list.append(mol)
        else:
            try:
                smiles_list.append(Chem.MolToSmiles(mol))
            except Exception:
                smiles_list.append(None)
    return smiles_list


def _score_docking_objective(
    objective_lower: str,
    smiles_list: list,
    return_normalized: bool,
    return_raw_scores: bool,
):
    http_base = _http_docking_base()
    if not http_base:
        raise RuntimeError(
            "Docking objective requested but oracle URL is not configured. "
            "Pass --oracle_url or set DOCKING_VINA_URL."
        )

    to_score = [(i, sm) for i, sm in enumerate(smiles_list) if sm]
    aff_map = {}
    if to_score:
        batch_smiles = [t[1] for t in to_score]
        client = DockingOracleClient(http_base, objective_lower)
        affs = client.predict(batch_smiles, raw_affinities=True)
        if len(affs) != len(batch_smiles):
            raise RuntimeError(
                f"Oracle length mismatch for {objective_lower}: "
                f"{len(batch_smiles)} smiles vs {len(affs)} scores"
            )
        for (idx, _smi), aff in zip(to_score, affs):
            try:
                af = float(aff)
            except Exception:
                af = 99.9
            aff_map[idx] = -1.0 if af >= 90.0 else af

    raw_scores_list = []
    for i, sm in enumerate(smiles_list):
        if not sm:
            raw_scores_list.append(0.0)
        elif i in aff_map:
            raw_scores_list.append(float(aff_map[i]))
        else:
            raw_scores_list.append(-1.0)

    scores = []
    for raw in raw_scores_list:
        if return_normalized:
            scores.append(0.0 if raw == -1.0 else float(normalize_docking_score(raw)))
        else:
            scores.append(float(raw))

    if return_raw_scores:
        return scores, raw_scores_list
    return scores


def get_scores(objective_name, mols, return_normalized=False, return_raw_scores=False):
    if not mols:
        return []

    smiles_list = _smiles_list_from_mols(mols)
    objective_lower = objective_name.lower()

    if objective_lower == "qed":
        oracle = _get_qed_oracle()
        scores = []
        for smiles in smiles_list:
            if smiles:
                try:
                    scores.append(float(oracle([smiles])[0]))
                except Exception:
                    scores.append(0.0)
            else:
                scores.append(0.0)

    elif objective_lower == "sa":
        oracle = _get_sa_oracle()
        scores = []
        for smiles in smiles_list:
            if smiles:
                try:
                    score = float(oracle([smiles])[0])
                    if return_normalized:
                        score = normalize_sa_score(score)
                    scores.append(score)
                except Exception:
                    scores.append(0.0)
            else:
                scores.append(0.0)

    elif objective_lower == "similarity":
        if not _LEAD_SMILES:
            raise ValueError("similarity objective requires set_lead_smiles() or --seed_mol")
        lead_mol = Chem.MolFromSmiles(_LEAD_SMILES)
        if lead_mol is None:
            raise ValueError(f"Invalid lead SMILES for similarity: {_LEAD_SMILES}")
        lead_fp = AllChem.GetMorganFingerprintAsBitVect(lead_mol, 2, 2048)
        scores = []
        for smiles in smiles_list:
            if not smiles:
                scores.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0.0)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            scores.append(float(DataStructs.TanimotoSimilarity(lead_fp, fp)))

    elif objective_lower in TARGET_CONFIGS:
        return _score_docking_objective(
            objective_lower,
            smiles_list,
            return_normalized,
            return_raw_scores,
        )

    elif objective_lower in ["gsk3b", "jnk3", "drd2"]:
        try:
            oracle = TDCOracle(name=objective_name.upper())
            scores = []
            for smiles in smiles_list:
                if smiles:
                    try:
                        scores.append(float(oracle([smiles])[0]))
                    except Exception:
                        scores.append(0.0)
                else:
                    scores.append(0.0)
        except Exception:
            scores = [0.0] * len(smiles_list)

    else:
        dock_names = ", ".join(sorted(TARGET_CONFIGS.keys()))
        raise ValueError(
            f"Unknown objective: {objective_name}. "
            f"Docking targets: {dock_names}. Other: qed, sa, similarity, gsk3b, jnk3, drd2"
        )

    if return_raw_scores:
        return scores, scores.copy()
    return scores
