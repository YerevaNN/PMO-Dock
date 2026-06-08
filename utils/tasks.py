"""
Shared task helpers for algorithm entrypoints (repo-root ``utils`` package).

Loads ``genetic_chemalactica/utils/tasks.py`` directly so importing this module
does not execute ``genetic_chemalactica.utils`` (which requires torch).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from benchmark.rewards import gaussian_modifier, select_sigma as _bench_select_sigma
from benchmark.tasks import lead_qed_sa_hit_constraints, task_name2constraints

_REPO = Path(__file__).resolve().parents[1]
_GC_TASKS_PATH = _REPO / "genetic_chemalactica" / "utils" / "tasks.py"
_spec = importlib.util.spec_from_file_location("_gc_tasks_module", _GC_TASKS_PATH)
assert _spec and _spec.loader
_gc_tasks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gc_tasks)

task_name2computer_names = _gc_tasks.task_name2computer_names
task_name2grpo_prompt = _gc_tasks.task_name2grpo_prompt
task_name2hit_ranges = _gc_tasks.task_name2hit_ranges
validate_task_name = _gc_tasks.validate_task_name
validate_reward_type = _gc_tasks.validate_reward_type
randfloat = _gc_tasks.randfloat
randint = _gc_tasks.randint
_gc_select_sigma = _gc_tasks.select_sigma


def select_sigma(prop_name: str):
    """Property sigma for reward shaping (benchmark keys first, then genetic naming)."""
    sigma = _bench_select_sigma(prop_name)
    if sigma is not None:
        return sigma
    return _gc_select_sigma(prop_name)


__all__ = [
    "gaussian_modifier",
    "lead_qed_sa_hit_constraints",
    "randfloat",
    "randint",
    "select_sigma",
    "task_name2computer_names",
    "task_name2constraints",
    "task_name2grpo_prompt",
    "task_name2hit_ranges",
]
