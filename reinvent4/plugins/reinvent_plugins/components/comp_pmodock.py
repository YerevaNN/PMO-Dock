"""PMO-Dock scoring component for REINVENT4.

Bridges REINVENT4's scoring subsystem to the shared ``benchmark`` oracles so that
REINVENT4 runs are directly comparable to saturn / genmol / genetic_gfn /
genetic_chemllama on the hit, lead and spec tasks.

Discovery
---------
This file lives in a *native namespace package* (no ``__init__.py`` anywhere under
``reinvent4/plugins/``) that merges with REINVENT4's own ``reinvent_plugins.components``.
Put ``$PROJECT_ROOT/reinvent4/plugins`` on ``PYTHONPATH`` and REINVENT4 picks it up
because the basename starts with ``comp_``.

Budget accounting
-----------------
REINVENT4 has no notion of an oracle budget: its only stop condition is ``max_steps``.
This component owns the budget instead, using the same semantics as
``saturn/oracles/oracle.py`` so the comparison is apples-to-apples:

* one oracle call == one *distinct canonical* SMILES actually scored,
* molecules REINVENT4 masks out before scoring (invalid, in-batch duplicate, filter
  rejects) cost nothing,
* repeats across steps are served from our own cache and cost nothing.

``no_cache = True`` disables REINVENT4's per-component cache so that this component
sees every unmasked SMILES and does the deduplication itself. That matters because
REINVENT4's cache is keyed on the raw (possibly randomized) SMILES string, while the
benchmark counts canonical SMILES; and because REINVENT4 rebuilds that cache at every
``[[stage]]`` boundary, which would re-score molecules in a curriculum run.

When the budget is exhausted the history is flushed and ``BudgetExhausted`` is raised. It
subclasses REINVENT4's ``StageInterruptedControlled``, so the run stops through REINVENT4's
own graceful path (checkpoint written, summary CSVs flushed, exit status 0).
"""

from __future__ import annotations

__all__ = ["PMODock"]

import logging
import os
import threading
from typing import List, Optional

import numpy as np
from pydantic.dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

from .add_tag import add_tag
from .component_results import ComponentResults

from benchmark.docking_oracle.docking import is_antitarget_receptor
from benchmark.docking_oracle.docking_vina_client import DockingOracleClient
from benchmark.rewards import hit_reward, select_sigma
from benchmark.spec_tasks import parse_geam_spec_target
from benchmark.synthesizability.sascorer import calculateScore
from benchmark.tasks import task_name2constraints

logger = logging.getLogger(__name__)

# Docking failure sentinel shared with benchmark.docking_oracle.
DOCKING_FAILURE = 99.9

# Lead similarity floor used when a run does not pass one. 0.6 is the setting this work
# is mainly interested in; the benchmark also names a 0.4 tier, and any other value is
# equally valid -- see --sim_thresholds in reinvent4/reinvent_lead_runner.py.
DEFAULT_SIM_THRESHOLD = 0.6

HISTORY_COLUMNS = [
    "oracle_calls",
    "smiles",
    "scaffold",
    "reward",
    "raw_vina",
    "qed",
    "raw_sa",
    "aggregated_reward",
]


try:
    # REINVENT4 catches this in reinvent/Reinvent.py::main, after Handler.__exit__ has
    # written the agent checkpoint -- the graceful stop path normally used for SIGUSR1.
    from reinvent.runmodes.handler import StageInterruptedControlled as _StopBase
except ImportError:
    import importlib.util

    if importlib.util.find_spec("reinvent") is not None:
        # REINVENT4 is installed but its handler is unreachable -- usually a missing
        # dependency inside reinvent.runmodes (torchvision is undeclared at v4.5.11).
        # Degrading to RuntimeError here would silently disable graceful termination and
        # turn budget exhaustion into a crash, so fail loudly instead.
        raise
    # No REINVENT4 at all: keep the component importable for unit tests.
    _StopBase = RuntimeError


class BudgetExhausted(_StopBase):
    """Raised once ``max_oracle_calls`` distinct molecules have been scored.

    Subclasses REINVENT4's ``StageInterruptedControlled`` so the run shuts down through
    its normal graceful path: checkpoint saved, summary CSVs flushed, exit status 0.
    """


@add_tag("__parameters")
@dataclass
class Parameters:
    """REINVENT4 passes every endpoint parameter as a list, one entry per endpoint."""

    target: List[str]
    budget: List[int]
    history_path: List[str]
    reward_type: List[str] = None
    lead_smiles: List[str] = None
    # Lead task only: the minimum Tanimoto similarity to the lead molecule. A run
    # parameter rather than a task constant -- the benchmark's 0.4/0.6 tiers are two
    # points on a continuum, and sweeping it is a legitimate experiment.
    sim_thr: List[float] = None
    oracle_url: List[str] = None
    vina_seed: List[int] = None


@add_tag("__component")
class PMODock:
    number_of_endpoints = 1

    def __init__(self, params: Parameters):
        # Single endpoint: the aggregated benchmark reward. Raw properties ride along as
        # metadata so REINVENT4's own summary CSV stays informative.
        self.target = params.target[0]
        self.budget = int(params.budget[0])
        self.history_path = params.history_path[0]
        self.reward_type = (params.reward_type or ["geam"])[0]
        self.lead_smiles = (params.lead_smiles or [""])[0]
        self.sim_thr = float((params.sim_thr or [DEFAULT_SIM_THRESHOLD])[0])
        self.vina_seed = int((params.vina_seed or [0])[0])

        if self.reward_type not in ("geam", "hit"):
            raise ValueError(f"reward_type must be 'geam' or 'hit', got {self.reward_type!r}")

        # Bypass REINVENT4's cache; we do our own canonical-SMILES accounting.
        self.no_cache = True

        spec = parse_geam_spec_target(self.target)
        if spec is not None:
            self.protein, self.antitarget = spec
        else:
            self.protein, self.antitarget = self.target, None

        url = (params.oracle_url or [""])[0] or os.environ.get("DOCKING_VINA_URL")
        if not url:
            raise ValueError(
                "REINVENT4 runs require the docking HTTP service: set DOCKING_VINA_URL or pass "
                "params.oracle_url. Inline DockingOracle needs OpenBabel/QuickVina, which are not "
                "installed in the reinvent4 env."
            )
        self.vina = DockingOracleClient(url, self.protein)
        self.vina_antitarget = (
            DockingOracleClient(url, self.antitarget) if self.antitarget else None
        )

        self.constraints = self._resolve_constraints()
        self.lead_fp = None
        if self.lead_smiles:
            lead_mol = Chem.MolFromSmiles(self.lead_smiles)
            if lead_mol is None:
                raise ValueError(f"Unparsable lead_smiles: {self.lead_smiles!r}")
            self.lead_fp = Chem.AllChem.GetMorganFingerprintAsBitVect(lead_mol, 2, 2048)

        # Oracle accounting. Guarded because REINVENT4 may score components in threads.
        self._lock = threading.Lock()
        self._scored: dict[str, float] = {}
        # Reserved by a call that is still waiting on the docking service. Only matters if
        # REINVENT4 is configured to score in parallel; without it two concurrent calls
        # could both treat the same molecule as new and spend the budget twice.
        self._in_flight: set[str] = set()
        self.calls = 0

        os.makedirs(os.path.dirname(os.path.abspath(self.history_path)) or ".", exist_ok=True)
        self._history_cols = list(HISTORY_COLUMNS)
        if self.antitarget:
            self._history_cols.insert(5, "raw_vina_antitarget")
        if self.lead_smiles:
            self._history_cols.append("raw_similarity")
        with open(self.history_path, "w", encoding="utf-8") as fh:
            fh.write(",".join(self._history_cols) + "\n")

        logger.info(
            "PMODock: target=%s antitarget=%s reward=%s budget=%d lead=%s",
            self.protein,
            self.antitarget,
            self.reward_type,
            self.budget,
            bool(self.lead_smiles),
        )
        if self.lead_smiles:
            logger.info("PMODock: similarity window [%.3f, 1.0]", self.sim_thr)

    def _resolve_constraints(self) -> dict:
        if self.antitarget:
            task_name = f"spec.{self.protein}_{self.antitarget}"
        else:
            # Lead uses the same per-protein docking/QED/SA windows as hit, matching
            # saturn's HITOracle; the similarity window is layered on below.
            task_name = f"hit.{self.protein}"

        try:
            constraints = dict(task_name2constraints(task_name))
        except KeyError:
            logger.warning("No constraints for %s; 'hit' reward falls back to defaults", task_name)
            constraints = {}

        if self.lead_smiles:
            # Built from the run's own threshold rather than looked up from a task table,
            # so any value is expressible, not just the two the benchmark happens to name.
            constraints["similarity_score"] = [self.sim_thr, 1.0]
        return constraints

    def __call__(self, smilies: List[str]) -> ComponentResults:
        canonical = [self._canonical(s) for s in smilies]

        with self._lock:
            if self.calls >= self.budget:
                self._raise_exhausted()
            fresh: list[str] = []
            for c in canonical:
                if c is None or c in self._scored or c in self._in_flight:
                    continue
                if self.calls + len(fresh) >= self.budget:
                    break
                fresh.append(c)
                self._in_flight.add(c)

        try:
            raw = self._score_batch(fresh) if fresh else {}
        except BaseException:
            with self._lock:
                self._in_flight.difference_update(fresh)
            raise

        with self._lock:
            for smi in fresh:
                self.calls += 1
                self._scored[smi] = raw[smi]["reward"]
                self._append_history(self.calls, smi, raw[smi])
            self._in_flight.difference_update(fresh)
            exhausted = self.calls >= self.budget

        scores = np.array(
            [self._scored.get(c, np.nan) if c is not None else np.nan for c in canonical],
            dtype=float,
        )

        if exhausted:
            # Flush first: the history is the experiment, the run is not.
            self._raise_exhausted()

        return ComponentResults([scores])

    @staticmethod
    def _canonical(smiles: str) -> Optional[str]:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None

    def _score_batch(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(s) for s in smiles]

        # benchmark convention: docking scores are negated affinities, higher is better.
        #
        # Failed docks: DockingOracleClient.predict already maps the 99.9 sentinel to 0.0,
        # so over HTTP -- the only path REINVENT4 uses -- a failure arrives as 0.0 and
        # scores as the worst possible dock. The DOCKING_FAILURE guards below therefore
        # never fire here; they are kept so the same code stays correct if a caller ever
        # passes raw affinities or an inline DockingOracle (which does return 99.9).
        # This matches saturn/oracles/docking/geam_oracle.py, which has the same shape.
        dock = -np.array(self.vina.predict(smiles, seed=self.vina_seed), dtype=float)
        dock[dock == -DOCKING_FAILURE] = DOCKING_FAILURE

        anti = None
        if self.vina_antitarget is not None:
            anti = -np.array(
                self.vina_antitarget.predict(smiles, seed=self.vina_seed), dtype=float
            )
            anti[anti == -DOCKING_FAILURE] = DOCKING_FAILURE

        qed = np.array([QED.qed(m) for m in mols], dtype=float)
        raw_sa = np.array([calculateScore(m) for m in mols], dtype=float)
        sa_reward = (10.0 - raw_sa) / 9.0

        similarity = None
        if self.lead_fp is not None:
            fps = [Chem.AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
            similarity = np.array(
                Chem.AllChem.DataStructs.BulkTanimotoSimilarity(self.lead_fp, fps), dtype=float
            )

        if self.reward_type == "geam":
            rewards = self._geam_rewards(dock, anti, qed, sa_reward, similarity)
        else:
            rewards = self._hit_rewards(dock, anti, qed, sa_reward, similarity)

        out = {}
        for i, smi in enumerate(smiles):
            row = {
                "reward": float(rewards[i]),
                "raw_vina": float(dock[i]),
                "qed": float(qed[i]),
                "raw_sa": float(raw_sa[i]),
            }
            if anti is not None:
                row["raw_vina_antitarget"] = float(anti[i])
            if similarity is not None:
                row["raw_similarity"] = float(similarity[i])
            out[smi] = row
        return out

    def _geam_rewards(self, dock, anti, qed, sa_reward, similarity) -> np.ndarray:
        """GEAM aggregation, matching saturn/oracles/docking/geam_oracle.py."""
        dock_norm = np.clip(np.clip(dock, 0, None), 0, 20) / 20.0
        dock_norm[dock == DOCKING_FAILURE] = 0.0

        if anti is not None and is_antitarget_receptor(self.antitarget or ""):
            anti_norm = np.clip(np.clip(anti, 0, None), 0, 20) / 20.0
            anti_norm[anti == DOCKING_FAILURE] = 0.0
            margin = np.clip(dock_norm - anti_norm, 0, 1)
            return margin * dock_norm * qed * sa_reward

        rewards = dock_norm * qed * sa_reward
        if similarity is not None:
            rewards = rewards * similarity
        return rewards

    def _hit_rewards(self, dock, anti, qed, sa_reward, similarity) -> np.ndarray:
        """Gaussian distance-to-constraint-window product, matching benchmark.rewards."""
        dock_sigma = select_sigma("DOCKING") or 0.1 * 20
        qed_sigma = select_sigma("QED") or 0.1 * 1
        sa_sigma = 0.1 * 1
        sim_sigma = select_sigma("SIMILAR") or 0.1 * 1

        dock_range = self.constraints.get("docking_score", [0.0, 20.0])
        qed_range = self.constraints.get("qed_score", [0.5, 1.0])
        sa_range = self.constraints.get("sa_score", [1.0, 5.0])
        anti_range = self.constraints.get("antitarget_docking_score", [0.0, 20.0])
        sim_range = self.constraints.get("similarity_score", [self.sim_thr, 1.0])

        rewards = []
        for i in range(len(dock)):
            dock_val = None if dock[i] == DOCKING_FAILURE else dock[i]
            measured = [dock_val]
            sigmas = [dock_sigma]
            ranges = [dock_range]
            if anti is not None:
                measured.append(None if anti[i] == DOCKING_FAILURE else anti[i])
                sigmas.append(dock_sigma)
                ranges.append(anti_range)
            measured += [qed[i], sa_reward[i]]
            sigmas += [qed_sigma, sa_sigma]
            ranges += [qed_range, sa_range]
            if similarity is not None:
                measured.append(similarity[i])
                sigmas.append(sim_sigma)
                ranges.append(sim_range)
            rewards.append(hit_reward(measured, sigmas, ranges, prod=True, avg=False))
        return np.array(rewards, dtype=float)

    @staticmethod
    def _scaffold(smiles: str) -> str:
        """Bemis-Murcko scaffold, same convention as saturn.utils.chemistry_utils."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        try:
            return Chem.MolToSmiles(GetScaffoldForMol(mol), canonical=True)
        except Exception:
            return ""

    def _append_history(self, call_no: int, smiles: str, row: dict) -> None:
        values = {
            "oracle_calls": call_no,
            "smiles": smiles,
            "scaffold": self._scaffold(smiles),
            "reward": row["reward"],
            "aggregated_reward": row["reward"],
            "raw_vina": row["raw_vina"],
            "qed": row["qed"],
            "raw_sa": row["raw_sa"],
            "raw_vina_antitarget": row.get("raw_vina_antitarget"),
            "raw_similarity": row.get("raw_similarity"),
        }
        line = ",".join(
            "" if values.get(c) is None else str(values.get(c)) for c in self._history_cols
        )
        with open(self.history_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

    def _raise_exhausted(self) -> None:
        logger.info(
            "PMODock: oracle budget exhausted (%d calls); history at %s",
            self.calls,
            self.history_path,
        )
        raise BudgetExhausted(f"{self.calls} oracle calls reached budget {self.budget}")
