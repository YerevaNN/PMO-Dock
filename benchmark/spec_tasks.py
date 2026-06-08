"""Helpers for 6nzp + antitarget specificity (spec) benchmark tasks."""

from __future__ import annotations

from benchmark.docking_oracle.docking import ANTITARGET_RECEPTORS, is_antitarget_receptor

SPEC_TARGET_RECEPTOR = "6nzp"

SPEC_TASK_CONSTRAINTS = {
    "qed_score": [0.40, 1.00],
    "sa_score": [1.00, 4.00],
    "docking_score": [10.67, 20.00],
    "antitarget_docking_score": [0.00, 20.00],
}


def list_antitarget_receptors() -> list[str]:
    return sorted(ANTITARGET_RECEPTORS)


def spec_task_name(antitarget: str) -> str:
    return f"spec.{SPEC_TARGET_RECEPTOR}_{antitarget}"


def geam_spec_target_name(antitarget: str) -> str:
    return f"{SPEC_TARGET_RECEPTOR}_{antitarget}"


def geam_spec_oracle_names() -> list[str]:
    return [geam_spec_target_name(at) for at in list_antitarget_receptors()]


def parse_geam_spec_target(name: str) -> tuple[str, str] | None:
    """Parse ``6nzp_7uyt`` -> (``6nzp``, ``7uyt``) when antitarget is known."""
    if not isinstance(name, str):
        return None
    prefix = f"{SPEC_TARGET_RECEPTOR}_"
    if not name.startswith(prefix):
        return None
    antitarget = name[len(prefix) :]
    if is_antitarget_receptor(antitarget):
        return SPEC_TARGET_RECEPTOR, antitarget
    return None


def parse_spec_task_name(task_name: str) -> tuple[str, str] | None:
    """Parse ``spec.6nzp_7uyt`` -> (``6nzp``, ``7uyt``) when antitarget is known."""
    if not isinstance(task_name, str):
        return None
    prefix = f"spec.{SPEC_TARGET_RECEPTOR}_"
    if not task_name.startswith(prefix):
        return None
    antitarget = task_name[len(prefix) :]
    if is_antitarget_receptor(antitarget):
        return SPEC_TARGET_RECEPTOR, antitarget
    return None


def is_spec_task_name(task_name: str) -> bool:
    return parse_spec_task_name(task_name) is not None


def is_geam_spec_target(name: str) -> bool:
    return parse_geam_spec_target(name) is not None
