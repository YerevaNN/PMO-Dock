from benchmark.spec_tasks import SPEC_TASK_CONSTRAINTS, is_spec_task_name

# Per-target docking floors shared by hit and lead (same s_min as in the paper appendix).
HIT_DOCKING_THRESHOLDS = {
    "parp1": [10.00, 20.00],
    "fa7": [8.50, 20.00],
    "5ht1b": [8.7845, 20.00],
    "braf": [10.30, 20.00],
    "jak2": [9.10, 20.00],
}

_LEAD_BY_SIM = {
    "04": {
        "qed_score": [0.60, 1.00],
        "sa_score": [1.00, 4.00],
        "similarity_score": [0.40, 1.00],
    },
    "06": {
        "qed_score": [0.60, 1.00],
        "sa_score": [1.00, 4.00],
        "similarity_score": [0.60, 1.00],
    },
}


def lead_constraints(target: str, sim: str) -> dict:
    """Full lead constraints for ``target`` at similarity tier ``sim`` (``'04'`` or ``'06'``).

    Includes the same per-target docking floor ``s_min`` used for hit generation.
    """
    if sim not in _LEAD_BY_SIM:
        raise KeyError(f"Unknown lead similarity tier: {sim!r} (expected '04' or '06')")
    if target not in HIT_DOCKING_THRESHOLDS:
        raise KeyError(f"Unknown lead target: {target!r}")
    out = dict(_LEAD_BY_SIM[sim])
    out["docking_score"] = list(HIT_DOCKING_THRESHOLDS[target])
    return out


def _parse_lead_task_name(task_name: str) -> tuple[str, str] | None:
    """Parse ``lead.braf_06_0`` -> ``('braf', '06')``."""
    if not isinstance(task_name, str) or not task_name.startswith("lead."):
        return None
    rest = task_name[len("lead.") :]
    if rest.startswith("sim_"):
        return None
    parts = rest.split("_")
    if len(parts) < 2:
        return None
    target, sim = parts[0], parts[1]
    if target in HIT_DOCKING_THRESHOLDS and sim in _LEAD_BY_SIM:
        return target, sim
    return None


def task_name2constraints(task_name):
    if is_spec_task_name(task_name):
        return dict(SPEC_TASK_CONSTRAINTS)
    parsed = _parse_lead_task_name(task_name)
    if parsed is not None:
        return lead_constraints(*parsed)
    if task_name in ("lead.sim_04", "lead.sim_06"):
        # Tier-only keys: QED/SA/sim. Call ``lead_constraints(target, sim)`` for docking floor.
        sim = task_name.rsplit("_", 1)[-1]
        return dict(_LEAD_BY_SIM[sim])
    return {
        "hit.parp1": {
            "qed_score": [0.50, 1.00],
            "sa_score": [1.00, 5.00],
            "docking_score": list(HIT_DOCKING_THRESHOLDS["parp1"]),
        },
        "hit.fa7": {
            "qed_score": [0.50, 1.00],
            "sa_score": [1.00, 5.00],
            "docking_score": list(HIT_DOCKING_THRESHOLDS["fa7"]),
        },
        "hit.5ht1b": {
            "qed_score": [0.50, 1.00],
            "sa_score": [1.00, 5.00],
            "docking_score": list(HIT_DOCKING_THRESHOLDS["5ht1b"]),
        },
        "hit.braf": {
            "qed_score": [0.50, 1.00],
            "sa_score": [1.00, 5.00],
            "docking_score": list(HIT_DOCKING_THRESHOLDS["braf"]),
        },
        "hit.jak2": {
            "qed_score": [0.50, 1.00],
            "sa_score": [1.00, 5.00],
            "docking_score": list(HIT_DOCKING_THRESHOLDS["jak2"]),
        },
    }[task_name]


def lead_qed_sa_hit_constraints(sim: str):
    """Lead docking curves: ``sim`` is ``'04'`` or ``'06'`` (similarity tier).

    Returns QED/SA/sim only. Prefer ``lead_constraints(target, sim)`` when the
    per-target docking floor is required.
    """
    return task_name2constraints(f"lead.sim_{sim}")
