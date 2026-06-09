from __future__ import annotations

import os
import secrets
import time


def generate_random_hash(n_bytes: int = 8) -> str:
    """Short, filesystem-friendly random id."""
    n_bytes = int(n_bytes)
    if n_bytes <= 0:
        n_bytes = 8
    return secrets.token_hex(n_bytes)


def _next_exp_dir(parent_dir: str, prefix: str = "exp") -> str:
    os.makedirs(parent_dir, exist_ok=True)
    i = 0
    while True:
        candidate = os.path.join(parent_dir, f"{prefix}-{i}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def get_log_dir(
    method: str,
    model_name: str = "",
    exp_name: str = "exp",
    suffix: str = "",
    task_type: str | None = None,
    **_: object,
) -> str:
    """
    Standard log layout used across runners.

    Logs under:
      ${OUT_DIR}/<method>/<task_type>/<YYYY-MM-DD>/<exp-N>/<model_name><suffix>

    ``OUT_DIR`` defaults to ``$PROJECT_ROOT/results`` (see ``.env_vars``).

    Callers may pass only ``model_name`` (legacy runners) or explicit ``task_type``.
    """
    if task_type is None:
        task_type = model_name or "default"
        model_name = ""
    out_dir = os.environ.get("OUT_DIR", ".")
    date_str = time.strftime("%Y-%m-%d")
    base = os.path.join(out_dir, str(method), str(task_type), date_str)
    exp_dir = _next_exp_dir(base, prefix=exp_name)
    tail = f"{model_name}{suffix}".strip()
    return os.path.join(exp_dir, tail) if tail else exp_dir


def get_job_dir(is_hparam_search: bool, cat: str = "jobs") -> str:
    """
    Directory for submitit / orchestration logs.

    Layout:
      ${OUT_DIR}/job_dirs/<cat>/<YYYY-MM-DD>-{hparam}/<hash>
    """
    out_dir = os.environ.get("OUT_DIR", ".")
    date_str = time.strftime("%Y-%m-%d")
    mode = "hparam" if is_hparam_search else ""
    base = os.path.join(out_dir, "job_dirs", str(cat), f"{date_str}-{mode}")
    return os.path.join(base, generate_random_hash(6))
