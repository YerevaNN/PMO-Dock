"""The docking oracle's semaphore sizing rule.

Two things are under test, both in ``benchmark/docking_oracle/docking.py``:

1. ``_allocated_cpus()`` must read the SLURM *allocation*, not the machine.
   ``multiprocessing.cpu_count()`` reports every core on the node, which SLURM
   does not give the job -- an 8-CPU job on a 224-core node sized its semaphore
   for 100+ concurrent docks. SLURM allocates by cpuset, which
   ``sched_getaffinity`` sees and ``cpu_count`` does not. This is a latent bug
   fix, not a tuning: every launcher in the tree sets
   ``MAX_CONCURRENT_DOCKING`` explicitly, so it is a bare ``DockingOracle``
   import (an analysis script, a notebook) that would hit it.

2. ``_default_max_concurrent()`` must be 1.5x allocated cores, not 0.9x.

Neither is visible in a run that sets ``MAX_CONCURRENT_DOCKING``, and both are
silent when wrong -- the node just thrashes -- so they are pinned here.

CPU-only and fast; ``benchmark/slurm/run_tests.slurm`` runs them.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("numpy")

from benchmark.docking_oracle import docking as dk


@pytest.fixture(autouse=True)
def _reset_semaphore():
    """The semaphore is a class attribute built once per interpreter."""
    dk.DockingOracle._docking_semaphore = None
    dk.DockingOracle._max_concurrent_docking = None
    yield
    dk.DockingOracle._docking_semaphore = None
    dk.DockingOracle._max_concurrent_docking = None


def _fake_affinity(monkeypatch, n):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(n)))


# --- _allocated_cpus -------------------------------------------------------


@pytest.mark.parametrize("n", [1, 8, 24, 32, 96])
def test_allocated_cpus_reads_the_affinity_mask(monkeypatch, n):
    _fake_affinity(monkeypatch, n)
    assert dk._allocated_cpus() == n


def test_allocated_cpus_ignores_the_machine_size(monkeypatch):
    """The job-390124 case: 8 allocated CPUs on a 224-core node."""
    _fake_affinity(monkeypatch, 8)
    monkeypatch.setattr(dk.multiprocessing, "cpu_count", lambda: 224)
    assert dk._allocated_cpus() == 8


def test_allocated_cpus_falls_back_to_cpu_count(monkeypatch):
    """Platforms without sched_getaffinity must still get a usable number."""

    def _raise(_pid):
        raise OSError("no affinity on this platform")

    monkeypatch.setattr(os, "sched_getaffinity", _raise)
    monkeypatch.setattr(dk.multiprocessing, "cpu_count", lambda: 12)
    assert dk._allocated_cpus() == 12


def test_allocated_cpus_is_never_zero(monkeypatch):
    """An empty mask would otherwise make the semaphore size 0 and deadlock."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set())
    assert dk._allocated_cpus() == 1


# --- _default_max_concurrent ----------------------------------------------


@pytest.mark.parametrize(
    "cpus,num_cpu_dock,expected",
    [
        (96, 1, 144),  # the standard service allocation: 1.5x cores
        (96, 2, 72),
        (32, 1, 48),
        (24, 1, 36),
        (8, 1, 12),  # job 390124: 12, not the 201 the old rule produced
        (8, 2, 6),
        (1, 4, 1),  # floored, never 0
    ],
)
def test_default_is_one_and_a_half_times_allocated_cores(
    monkeypatch, cpus, num_cpu_dock, expected
):
    _fake_affinity(monkeypatch, cpus)
    assert dk._default_max_concurrent(num_cpu_dock) == expected


def test_default_beats_the_old_zero_point_nine_rule(monkeypatch):
    """Pins the direction of the change, not just the arithmetic."""
    _fake_affinity(monkeypatch, 96)
    old = max(1, int(96 / 1 * 0.9))
    assert dk._default_max_concurrent(1) > old


# --- _init_semaphore -------------------------------------------------------


class _StubOracle:
    """Enough of a DockingOracle to run _init_semaphore.

    Constructing the real thing needs a grids directory and a real receptor,
    neither of which belongs in a unit test.
    """

    num_cpu_dock = 1

    _init_semaphore = dk.DockingOracle._init_semaphore


def test_explicit_env_var_wins_over_the_default(monkeypatch):
    _fake_affinity(monkeypatch, 96)
    monkeypatch.setenv("MAX_CONCURRENT_DOCKING", "7")
    _StubOracle()._init_semaphore()
    assert dk.DockingOracle._max_concurrent_docking == 7


def test_unset_env_var_uses_the_allocation_based_default(monkeypatch):
    _fake_affinity(monkeypatch, 8)
    monkeypatch.setattr(dk.multiprocessing, "cpu_count", lambda: 224)
    monkeypatch.delenv("MAX_CONCURRENT_DOCKING", raising=False)
    _StubOracle()._init_semaphore()
    assert dk.DockingOracle._max_concurrent_docking == 12


def test_semaphore_is_built_once_per_interpreter(monkeypatch):
    """Second call must not resize: five seed processes each build their own,
    but within one process the first value is the one that throttles."""
    _fake_affinity(monkeypatch, 8)
    monkeypatch.delenv("MAX_CONCURRENT_DOCKING", raising=False)
    _StubOracle()._init_semaphore()
    first = dk.DockingOracle._docking_semaphore

    monkeypatch.setenv("MAX_CONCURRENT_DOCKING", "999")
    _StubOracle()._init_semaphore()
    assert dk.DockingOracle._docking_semaphore is first
    assert dk.DockingOracle._max_concurrent_docking == 12
