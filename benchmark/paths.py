from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """
    Canonical project root for all file paths.

    Public runs should set PROJECT_ROOT explicitly. For developer convenience,
    we fall back to inferring the repo root from this file location.
    """
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # This file lives in <repo>/benchmark/paths.py
    return Path(__file__).resolve().parents[1]


def resolve_from_project_root(*parts: str | os.PathLike) -> Path:
    """Join path components under PROJECT_ROOT."""
    return get_project_root().joinpath(*parts)


def expand_path_string(s: str) -> str:
    """Expand ${VARS} and ~ in a path-like string."""
    return os.path.expanduser(os.path.expandvars(s))


def expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in nested dict/list config objects."""
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [expand_env_vars(v) for v in obj]
        return type(obj)(out) if isinstance(obj, tuple) else out
    if isinstance(obj, str):
        return expand_path_string(obj)
    return obj


def resolve_maybe_relative(path: str, *, base_dir: str | os.PathLike) -> str:
    """If `path` is relative, resolve it under `base_dir`. Always expands env vars."""
    p = Path(expand_path_string(path))
    if not p.is_absolute():
        p = Path(base_dir) / p
    return str(p.resolve())
