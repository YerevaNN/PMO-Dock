from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator


class BenchmarkTimer:
    """Lightweight phase timer for oracle / genetic runs (optional)."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        _ = name
        yield
