"""Budget accounting for the PMODock REINVENT4 component.

REINVENT4 has no oracle budget of its own, so this component is the only thing standing
between a run and an unbounded number of docking calls. These tests pin the semantics to
``saturn/oracles/oracle.py``: one call per *distinct canonical* SMILES, repeats free.

Run with:  PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/reinvent4/plugins pytest reinvent4/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "reinvent4", "plugins")):
    if path not in sys.path:
        sys.path.insert(0, path)

from reinvent_plugins.components import comp_pmodock  # noqa: E402
from reinvent_plugins.components.comp_pmodock import BudgetExhausted, PMODock  # noqa: E402

# Distinct, valid, canonically different molecules.
MOLECULES = [
    "CCO",
    "CCCO",
    "CCCCO",
    "c1ccccc1",
    "c1ccncc1",
    "CC(=O)O",
    "CCN",
    "CCCN",
]


class FakeDockingClient:
    """Stands in for DockingOracleClient and counts how many molecules it was asked about."""

    instances: list["FakeDockingClient"] = []

    def __init__(self, url, target, *args, **kwargs):
        self.target = target
        self.seen: list[str] = []
        FakeDockingClient.instances.append(self)

    def predict(self, smiles_list, seed=0, raw_affinities=False):
        self.seen.extend(smiles_list)
        # benchmark convention: the client returns affinities, the component negates them.
        return [-8.5] * len(smiles_list)


@pytest.fixture(autouse=True)
def fake_docking(monkeypatch):
    FakeDockingClient.instances = []
    monkeypatch.setattr(comp_pmodock, "DockingOracleClient", FakeDockingClient)


def make_component(tmp_path, budget, **overrides):
    params = comp_pmodock.Parameters(
        target=["parp1"],
        budget=[budget],
        history_path=[str(tmp_path / "history.csv")],
        reward_type=[overrides.get("reward_type", "geam")],
        lead_smiles=[overrides.get("lead_smiles", "")],
        sim_thr=[overrides.get("sim_thr", 0.6)],
        oracle_url=["http://fake-oracle:5050"],
        vina_seed=[0],
    )
    return PMODock(params)


def history_rows(component) -> list[str]:
    with open(component.history_path, "r", encoding="utf-8") as fh:
        return [line for line in fh.read().splitlines()[1:] if line]


def test_distinct_molecules_each_cost_one_call(tmp_path):
    component = make_component(tmp_path, budget=100)

    component(MOLECULES[:4])
    assert component.calls == 4

    component(MOLECULES[4:])
    assert component.calls == len(MOLECULES)
    assert len(history_rows(component)) == len(MOLECULES)


def test_repeats_across_calls_are_free(tmp_path):
    component = make_component(tmp_path, budget=100)

    component(MOLECULES[:4])
    component(MOLECULES[:4])
    component(MOLECULES[:4])

    assert component.calls == 4
    assert len(history_rows(component)) == 4
    # The docking service saw each molecule exactly once.
    assert len(FakeDockingClient.instances[0].seen) == 4


def test_in_batch_duplicates_and_noncanonical_forms_are_free(tmp_path):
    component = make_component(tmp_path, budget=100)

    # "OCC" and "C(C)O" are the same molecule as "CCO" once canonicalized.
    component(["CCO", "OCC", "C(C)O", "CCCO"])

    assert component.calls == 2
    assert len(FakeDockingClient.instances[0].seen) == 2


def test_budget_is_never_exceeded_and_stops_the_run(tmp_path):
    budget = 5
    component = make_component(tmp_path, budget=budget)

    # One batch larger than the whole budget.
    with pytest.raises(BudgetExhausted):
        component(MOLECULES)

    assert component.calls == budget
    assert len(history_rows(component)) == budget
    assert len(FakeDockingClient.instances[0].seen) == budget


def test_further_calls_after_exhaustion_still_raise(tmp_path):
    component = make_component(tmp_path, budget=2)

    with pytest.raises(BudgetExhausted):
        component(MOLECULES[:4])
    with pytest.raises(BudgetExhausted):
        component(MOLECULES[:1])

    assert component.calls == 2


def test_scores_align_with_input_order(tmp_path):
    component = make_component(tmp_path, budget=100)

    smiles = ["CCO", "not-a-molecule", "CCCO", "CCO"]
    results = component(smiles)
    scores = results.scores[0]

    assert len(scores) == len(smiles)
    assert np.isnan(scores[1])  # unparsable -> NaN, never a zero that looks like a score
    assert scores[0] == scores[3]  # same molecule, same score
    assert not np.isnan(scores[0]) and not np.isnan(scores[2])
    assert component.calls == 2


def test_lead_mode_scores_similarity_and_logs_it(tmp_path):
    component = make_component(tmp_path, budget=100, lead_smiles="CCCO", sim_thr=0.6)

    assert "raw_similarity" in component._history_cols
    component(["CCO", "CCCO"])

    rows = history_rows(component)
    assert len(rows) == 2
    # The seed molecule scored against itself has similarity 1.0.
    self_row = next(r for r in rows if r.split(",")[1] == "CCCO")
    assert float(self_row.split(",")[-1]) == pytest.approx(1.0)


def test_spec_mode_adds_antitarget_column_and_second_oracle(tmp_path):
    component = make_component(tmp_path, budget=100)
    assert component.antitarget is None

    spec = comp_pmodock.Parameters(
        target=["6nzp_7uyt"],
        budget=[100],
        history_path=[str(tmp_path / "spec_history.csv")],
        reward_type=["geam"],
        oracle_url=["http://fake-oracle:5050"],
        vina_seed=[0],
    )
    spec_component = PMODock(spec)

    assert spec_component.protein == "6nzp"
    assert spec_component.antitarget == "7uyt"
    assert "raw_vina_antitarget" in spec_component._history_cols

    spec_component(MOLECULES[:3])
    assert spec_component.calls == 3
    # Target and antitarget are separate docking clients, both consulted.
    targets = [c.target for c in FakeDockingClient.instances if c.seen]
    assert "6nzp" in targets and "7uyt" in targets


def test_requires_docking_service(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCKING_VINA_URL", raising=False)
    params = comp_pmodock.Parameters(
        target=["parp1"],
        budget=[10],
        history_path=[str(tmp_path / "h.csv")],
        oracle_url=[""],
    )
    with pytest.raises(ValueError, match="DOCKING_VINA_URL"):
        PMODock(params)


def test_similarity_window_follows_the_run_threshold(tmp_path):
    """The similarity floor is whatever the run passed, not a value from a task table.

    The benchmark names a 0.4 and a 0.6 tier; a run may legitimately ask for neither.
    """
    component = make_component(tmp_path, budget=10, lead_smiles="CCCO", sim_thr=0.75)
    assert component.constraints["similarity_score"] == [0.75, 1.0]


def test_similarity_threshold_defaults_to_06(tmp_path):
    """Unspecified means 0.6 -- the setting this work is mainly interested in."""
    params = comp_pmodock.Parameters(
        target=["parp1"],
        budget=[10],
        history_path=[str(tmp_path / "history.csv")],
        lead_smiles=["CCCO"],
        oracle_url=["http://fake-oracle:5050"],
    )
    component = PMODock(params)
    assert component.sim_thr == 0.6
    assert component.constraints["similarity_score"] == [0.6, 1.0]


def test_hit_reward_uses_the_run_threshold(tmp_path):
    """A molecule inside a loose window but outside a strict one must score differently."""
    loose = make_component(tmp_path / "loose", budget=10, lead_smiles="CCCO",
                           sim_thr=0.1, reward_type="hit")
    strict = make_component(tmp_path / "strict", budget=10, lead_smiles="CCCO",
                            sim_thr=0.9, reward_type="hit")
    assert loose.constraints["similarity_score"][0] == 0.1
    assert strict.constraints["similarity_score"][0] == 0.9
