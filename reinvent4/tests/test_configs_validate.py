"""Validate the shipped JSON configs against REINVENT4's own pydantic schema.

Catches structural config drift -- a renamed key, a wrong nesting level, a value outside
its allowed range -- without burning a GPU job to find out. Skipped when REINVENT4 is not
installed, so the suite still runs anywhere.

Run inside the reinvent4 env:
    PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/reinvent4/plugins pytest reinvent4/tests -q
"""

from __future__ import annotations

import json
import os

import pytest

reinvent_rl = pytest.importorskip(
    "reinvent.runmodes.RL.validation",
    reason="REINVENT4 not installed; config validation needs the upstream schema",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "reinvent4", "configs")
CONFIGS = ["hit.json", "spec.json", "lead_m2m.json", "lead_libinvent.json"]


def load(name: str) -> dict:
    with open(os.path.join(CONFIG_DIR, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("name", CONFIGS)
def test_config_matches_upstream_top_level_schema(name):
    """Each config parses as a valid ReinventConfig -- the gate Reinvent.py applies first.

    This is the check that matters: ``ReinventConfig`` sets ``extra="forbid"``, so any
    top-level key upstream does not declare aborts the run before anything else happens.
    An earlier version of this test validated only ``RLConfig`` and hand-stripped the keys
    it assumed sat above it, ``seed`` among them -- which is exactly the key that turned
    out to be forbidden. Validate what upstream validates, not what we think it validates.
    """
    from reinvent.validation import ReinventConfig

    ReinventConfig(**load(name))


@pytest.mark.parametrize("name", CONFIGS)
def test_config_sections_match_rl_schema(name):
    """The run-mode sections parse as a valid RLConfig."""
    config = load(name)
    sections = {k: v for k, v in config.items() if k not in ("run_type", "device", "tb_logdir")}
    reinvent_rl.RLConfig(**sections)


@pytest.mark.parametrize("name", CONFIGS)
def test_config_carries_no_seed_key(name):
    """`seed` must not appear in a config: ReinventConfig forbids it.

    The seed is applied by ``reinvent4/reinvent_launch.py`` instead, because REINVENT4
    v4.5.11's own ``--seed`` is a no-op. If this ever starts failing because upstream
    added a `seed` field, the launcher can go.
    """
    assert "seed" not in load(name)


def test_launcher_seeds_for_real(monkeypatch):
    """The launcher must actually call set_seed with the seed it was given.

    Upstream accepts --seed and silently ignores it; that is the bug this module exists to
    work around, so the workaround itself needs a test.
    """
    import reinvent.utils.helpers as helpers

    from reinvent4 import reinvent_launch

    seen = {}
    monkeypatch.setattr(helpers, "set_seed", lambda s: seen.setdefault("seed", s))
    monkeypatch.setattr("reinvent.Reinvent.main_script", lambda: seen.setdefault("ran", True))

    reinvent_launch.main(["--seed", "7", "/nonexistent/config.json"])

    assert seen["seed"] == 7
    assert seen["ran"] is True


@pytest.mark.parametrize("name", CONFIGS)
def test_scoring_component_is_pmodock(name):
    """The only scoring component is ours -- REINVENT4's aggregator must stay a no-op."""
    config = load(name)
    components = config["stage"][0]["scoring"]["component"]
    assert len(components) == 1, "a second component would make REINVENT4 aggregate"
    assert list(components[0]) == ["PMODock"]
    endpoints = components[0]["PMODock"]["endpoint"]
    assert len(endpoints) == 1
    assert endpoints[0]["weight"] == 1.0


def test_seeded_configs_declare_a_smiles_file():
    """Mol2Mol and LibInvent need a seed structure; the de novo configs must not have one."""
    for name in ("lead_m2m.json", "lead_libinvent.json"):
        assert "smiles_file" in load(name)["parameters"], name
    for name in ("hit.json", "spec.json"):
        assert "smiles_file" not in load(name)["parameters"], name


def test_inception_only_on_de_novo_configs():
    """Upstream guards inception on model_type == 'Reinvent' (setup/inception.py)."""
    for name in ("hit.json", "spec.json"):
        assert "inception" in load(name), name
    for name in ("lead_m2m.json", "lead_libinvent.json"):
        assert "inception" not in load(name), name


def test_priors_resolve():
    """Every prior the configs name exists on disk via the registry or as a path."""
    from reinvent.utils.prior_registry import prior_registry

    for name in CONFIGS:
        params = load(name)["parameters"]
        for key in ("prior_file", "agent_file"):
            ref = params[key]
            path = prior_registry.get(ref, ref) if ref.startswith(".") else ref
            assert os.path.exists(path), f"{name}: {key}={ref} -> {path} missing"


@pytest.mark.parametrize("sim_thr", [0.0, 0.4, 0.5, 0.6, 0.9, 1.0])
def test_lead_runner_m2m_prior_exists_for_any_threshold(sim_thr):
    """Every similarity floor a run may ask for resolves to a prior that is on disk.

    Parameterised across the cutoff rather than over two fixed tiers: the threshold is a
    free run parameter now, so a value the benchmark never names must still work.
    """
    from reinvent.utils.prior_registry import prior_registry

    from reinvent4.reinvent_lead_runner import m2m_prior_for

    ref = m2m_prior_for(sim_thr)
    assert ref in prior_registry, f"sim_thr {sim_thr}: {ref} not in registry"
    assert os.path.exists(prior_registry[ref]), f"sim_thr {sim_thr}: {prior_registry[ref]} missing"
