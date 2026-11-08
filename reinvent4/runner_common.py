"""Shared plumbing for the REINVENT4 runners.

Mirrors the conventions established by ``saturn/saturn_hit_runner.py`` so that REINVENT4
experiments land in the same directory layout and accept the same CLI flags.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REINVENT4_DIR = os.path.join(PROJECT_ROOT, "reinvent4")
PLUGIN_DIR = os.path.join(REINVENT4_DIR, "plugins")

# Safety net only: the PMODock component is what actually ends a run, once the oracle
# budget is spent. This bounds the stage in case the agent collapses onto molecules that
# are all cached or invalid and therefore never consume budget.
MAX_STEPS_SLACK = 20


def extract_hparam_combinations(hparam_config: dict) -> list[tuple[str, list]]:
    """Flatten a nested hparam spec into ``[(dotted.path[idx], [values...]), ...]``."""

    def _extract(obj: Any, prefix: str = "", result: list | None = None) -> list:
        if result is None:
            result = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, list) and value:
                    if not isinstance(value[0], dict):
                        result.append((path, value))
                    else:
                        for i, item in enumerate(value):
                            _extract(item, f"{path}[{i}]", result)
                elif isinstance(value, dict):
                    _extract(value, path, result)
                else:
                    result.append((path, [value]))
        return result

    return _extract(hparam_config)


def set_nested_value(config: dict, path: str, value: Any) -> None:
    """Set ``config`` at a dotted path with optional ``[i]`` list indices."""
    parts = []
    for key, idx in re.findall(r"(\w+)|\[(\d+)\]", path):
        parts.append(("index", int(idx)) if idx else ("key", key))

    current = config
    for i, (kind, val) in enumerate(parts[:-1]):
        nxt_is_index = i + 1 < len(parts) and parts[i + 1][0] == "index"
        if kind == "key":
            if val not in current:
                current[val] = [] if nxt_is_index else {}
            current = current[val]
        else:
            while len(current) <= val:
                current.append([] if nxt_is_index else {})
            current = current[val]

    kind, val = parts[-1]
    if kind == "index":
        while len(current) <= val:
            current.append({})
        current[val] = value
    else:
        current[val] = value


def expand_hparams(base_config: dict, hparam_path: str) -> list[dict]:
    with open(hparam_path, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    pairs = extract_hparam_combinations(spec)
    paths = [p for p, _ in pairs]
    value_lists = [v for _, v in pairs]

    configs = []
    for combo in itertools.product(*value_lists):
        cfg = copy.deepcopy(base_config)
        for path, value in zip(paths, combo):
            set_nested_value(cfg, path, value)
        configs.append(cfg)
    return configs


def pmodock_params(config: dict) -> dict:
    """Return the mutable PMODock endpoint params dict of stage 0."""
    return config["stage"][0]["scoring"]["component"][0]["PMODock"]["endpoint"][0]["params"]


def set_budget(config: dict, budget: int) -> None:
    """Set the oracle budget and bound ``max_steps`` as a safety net."""
    for stage in config["stage"]:
        for component in stage["scoring"]["component"]:
            for endpoint in component["PMODock"]["endpoint"]:
                endpoint["params"]["budget"] = budget
    batch_size = max(1, int(config["parameters"].get("batch_size", 64)))
    config["stage"][-1]["max_steps"] = math.ceil(budget / batch_size) * MAX_STEPS_SLACK


def infer_n_gpus(explicit: int | None) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return 1
    return max(1, len([p for p in cvd.split(",") if p.strip()]))


def round_robin_device(i: int, n_gpus: int) -> str:
    return f"cuda:{i % max(1, n_gpus)}"


def write_config(config: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    return path


def run_one(cfg_path: str, seed: int) -> int:
    """Run a single REINVENT4 job at a given seed.

    Launched through ``reinvent4.reinvent_launch`` rather than ``reinvent.Reinvent``:
    REINVENT4 v4.5.11's own ``--seed`` is inert, and the config key that would feed it is
    rejected by ``ReinventConfig(extra="forbid")``. See that module for the details.
    """
    env = os.environ.copy()
    pythonpath = [PLUGIN_DIR, PROJECT_ROOT]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    # set_seed() assigns PYTHONHASHSEED too, but does it after the interpreter has already
    # started, where it has no effect. Setting it on the child's environment makes it real.
    env["PYTHONHASHSEED"] = str(seed)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "reinvent4.reinvent_launch",
            "--seed",
            str(seed),
            os.path.abspath(cfg_path),
        ],
        cwd=os.path.dirname(os.path.abspath(cfg_path)),
        env=env,
    )
    proc.wait()
    return proc.returncode


def run_all(
    jobs: list[tuple[str, int]], max_workers: int | None = None, stagger: int = 10
) -> int:
    """Run every (config, seed) job. Returns the number that failed."""
    job_start = time.time()
    if max_workers is None:
        max_workers = min(len(jobs), 5)

    logging.info("Starting %d REINVENT4 jobs with %d parallel workers", len(jobs), max_workers)

    def _run(cfg_path: str, seed: int):
        try:
            rc = run_one(cfg_path, seed)
            if rc == 0:
                return cfg_path, None
            logging.error("%s: exit code %s", cfg_path, rc)
            return cfg_path, f"Exit code: {rc}"
        except Exception as exc:  # noqa: BLE001 - one failed job must not kill the sweep
            logging.error("%s: %s", cfg_path, exc)
            return cfg_path, str(exc)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, (cfg_path, seed) in enumerate(jobs):
            if i > 0:
                time.sleep(stagger)
            futures[executor.submit(_run, cfg_path, seed)] = cfg_path
        for future in as_completed(futures):
            results.append(future.result())

    successful = sum(1 for _, err in results if err is None)
    failed = len(results) - successful
    logging.info(
        "Complete: %d successful, %d failed (%.1fmin)",
        successful,
        failed,
        (time.time() - job_start) / 60,
    )
    # Returned, not swallowed: a sweep in which every job died must not look like a green
    # SLURM job with no results.
    return failed
