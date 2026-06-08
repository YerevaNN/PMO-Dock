"""Reward helpers re-exported for Saturn / genetic_chemalactica oracles."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from benchmark.rewards import (
    compute_geam_reward,
    gaussian_modifier,
    hit_reward,
    select_sigma,
)

_REPO = Path(__file__).resolve().parents[1]
_GC_REWARDS_PATH = _REPO / "genetic_chemalactica" / "utils" / "rewards.py"
_spec = importlib.util.spec_from_file_location("_gc_rewards_module", _GC_REWARDS_PATH)
assert _spec and _spec.loader
_gc_rewards = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gc_rewards)

compute_gaussian_err = _gc_rewards.compute_gaussian_err
hit_docking_score_reward = _gc_rewards.hit_docking_score_reward
hit_similarity_reward = _gc_rewards.hit_similarity_reward
hit_spec_reward = _gc_rewards.hit_spec_reward

__all__ = [
    "compute_gaussian_err",
    "compute_geam_reward",
    "gaussian_modifier",
    "hit_docking_score_reward",
    "hit_reward",
    "hit_similarity_reward",
    "hit_spec_reward",
    "select_sigma",
]
