#!/usr/bin/env python
"""Run REINVENT4 on PMO-Dock lead tasks with a *seeded* generator.

This is the part of REINVENT4 that no other method in the benchmark has. The lead task
asks for molecules that stay within a Tanimoto radius of a known active while docking
better; REINVENT4's Mol2Mol and LibInvent priors take that seed as input, so the
constraint is carried by the model rather than by the reward alone.

Two generators:

* ``--generator mol2mol`` -- the similarity transformer. ``--sim_thresholds`` sets the
  similarity floor(s) to sweep and, by default, picks the prior trained for that regime
  (>= 0.5 -> ``.m2m_high``, below -> ``.m2m_medium``; override with ``--m2m_prior``).
  The seed SMILES is written to
  ``seed.smi`` and resampled every RL step (``learning.py`` holds ``input_smilies`` fixed),
  so one seed yields ``batch_size`` analogues per step.
* ``--generator libinvent`` -- R-group decoration. The seed's Bemis-Murcko scaffold with
  attachment points is written to ``scaffold.smi``; the shared core makes the similarity
  constraint structural.

Directory layout matches ``saturn/saturn_lead_runner.py``, with the similarity
threshold as the outermost sweep level because it changes the task, not just a
hyper-parameter: ``<log_dir>/sim_thr-T/start_mol_idx-K/<protein>/seed-N/``. The
``sim_thr-`` naming follows ``genmol/genmol_lead_runner.py``.

Example:

    python reinvent4/reinvent_lead_runner.py \
      --config_file reinvent4/configs/lead_m2m.json --generator mol2mol \
      --oracle_name parp1 --start_mol_idx 0 1 2 --seeds 0 1 2 \
      --sim_thresholds 0.6 --max_oracle_calls 1000
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem  # noqa: E402
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol  # noqa: E402

from benchmark.actives_loader import lead_seed_smiles  # noqa: E402
from benchmark.experiment_utils import get_job_dir, get_log_dir  # noqa: E402
from reinvent4.runner_common import (  # noqa: E402
    expand_hparams,
    infer_n_gpus,
    pmodock_params,
    round_robin_device,
    run_all,
    set_budget,
    write_config,
)

# Mol2Mol priors are trained per similarity regime. The cut is at 0.5 because upstream
# builds `.m2m_high` from pairs above roughly that similarity and `.m2m_medium` from the
# band below it; `--m2m_prior` overrides when a run wants to pair them differently.
M2M_HIGH_PRIOR = ".m2m_high"
M2M_MEDIUM_PRIOR = ".m2m_medium"
M2M_PRIOR_CUTOFF = 0.5


def m2m_prior_for(sim_thr: float) -> str:
    """Mol2Mol prior matching a similarity floor."""
    return M2M_HIGH_PRIOR if sim_thr >= M2M_PRIOR_CUTOFF else M2M_MEDIUM_PRIOR


def scaffold_with_attachment_points(smiles: str) -> str:
    """Bemis-Murcko scaffold with ``*`` where the stripped R-groups were attached.

    LibInvent expects a scaffold whose attachment points are explicit. Atoms that carried
    a side chain in the parent keep one open valence each.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unparsable lead SMILES: {smiles!r}")
    scaffold = GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        raise ValueError(f"No Murcko scaffold for {smiles!r}")

    match = mol.GetSubstructMatch(scaffold)
    if not match:
        raise ValueError(f"Scaffold does not match parent for {smiles!r}")
    in_scaffold = set(match)

    # A scaffold atom is an attachment point if the parent attached something to it
    # that the scaffold does not contain.
    attachment_sites = []
    for scaffold_idx, parent_idx in enumerate(match):
        parent_atom = mol.GetAtomWithIdx(parent_idx)
        for neighbor in parent_atom.GetNeighbors():
            if neighbor.GetIdx() not in in_scaffold:
                attachment_sites.append(scaffold_idx)
                break

    if not attachment_sites:
        raise ValueError(
            f"Lead {smiles!r} has no R-groups to decorate; use --generator mol2mol for it"
        )

    editable = Chem.RWMol(scaffold)
    for idx in attachment_sites:
        dummy = editable.AddAtom(Chem.Atom(0))
        editable.AddBond(idx, dummy, Chem.BondType.SINGLE)

    decorated = editable.GetMol()
    Chem.SanitizeMol(decorated)
    return Chem.MolToSmiles(decorated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--generator", choices=["mol2mol", "libinvent"], default="mol2mol")
    parser.add_argument("--oracle_name", nargs="+", required=True, type=str)
    parser.add_argument("--start_mol_idx", nargs="+", required=True, type=int)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument(
        "--sim_thresholds",
        nargs="+",
        type=float,
        default=[0.6],
        help="Minimum Tanimoto similarity to the lead; one run per value (default: 0.6).",
    )
    parser.add_argument(
        "--m2m_prior",
        type=str,
        default=None,
        help="Override the Mol2Mol prior otherwise chosen from the similarity threshold.",
    )
    parser.add_argument("--max_oracle_calls", required=True, type=int)
    parser.add_argument("--reward_type", type=str, default="geam", choices=["geam", "hit"])
    parser.add_argument("--hparam_config", type=str, default=None)
    parser.add_argument("--oracle_url", type=str, default=None)
    parser.add_argument("--n_gpus", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=15)
    parser.add_argument("--search_range", type=int, nargs=2, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    oracle_url = args.oracle_url or os.environ.get("DOCKING_VINA_URL")
    if not oracle_url:
        raise SystemExit(
            "Set --oracle_url or DOCKING_VINA_URL. Start the service with "
            "./benchmark/docking_oracle/start_oracle.sh"
        )

    project_root = os.environ.get("PROJECT_ROOT") or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    config_path = args.config_file
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        base_config = json.load(fh)

    if args.hparam_config:
        hparam_path = args.hparam_config
        if not os.path.isabs(hparam_path):
            hparam_path = os.path.join(project_root, "reinvent4", hparam_path)
        configs = expand_hparams(base_config, hparam_path)
    else:
        configs = [base_config]

    if args.search_range is not None:
        configs = configs[args.search_range[0] : args.search_range[1]]

    get_job_dir(args.hparam_config is not None, cat="reinvent4-lead")
    n_gpus = infer_n_gpus(args.n_gpus)

    model_name = os.path.splitext(os.path.basename(args.config_file))[0]
    # (config path, seed): the seed cannot live in the config -- REINVENT4's
    # ReinventConfig forbids the key. See reinvent4/reinvent_launch.py.
    jobs: list[tuple[str, int]] = []
    device_counter = 0

    for config in configs:
        log_dir = get_log_dir(
            method="reinvent4",
            model_name=model_name,
            exp_name="exp",
            suffix="-hparam" if args.hparam_config else "",
        )
        os.makedirs(log_dir, exist_ok=True)

        for sim_thr in args.sim_thresholds:
            if not 0.0 <= sim_thr <= 1.0:
                raise SystemExit(f"--sim_thresholds must lie in [0, 1], got {sim_thr}")
            thr_dir = os.path.join(log_dir, f"sim_thr-{sim_thr}")
            os.makedirs(thr_dir, exist_ok=True)

            for start_mol_idx in args.start_mol_idx:
                idx_dir = os.path.join(thr_dir, f"start_mol_idx-{start_mol_idx}")
                os.makedirs(idx_dir, exist_ok=True)

                for oracle in args.oracle_name:
                    lead = lead_seed_smiles(oracle, start_mol_idx)
                    oracle_dir = os.path.join(idx_dir, oracle)
                    os.makedirs(oracle_dir, exist_ok=True)

                    for seed in args.seeds:
                        seed_dir = os.path.join(oracle_dir, f"seed-{seed}")
                        os.makedirs(seed_dir, exist_ok=True)

                        if args.generator == "mol2mol":
                            smiles_name = "seed.smi"
                            seed_input = lead
                        else:
                            smiles_name = "scaffold.smi"
                            seed_input = scaffold_with_attachment_points(lead)
                        smiles_path = os.path.join(seed_dir, smiles_name)
                        with open(smiles_path, "w", encoding="utf-8") as fh:
                            fh.write(seed_input + "\n")

                        cfg = copy.deepcopy(config)
                        if args.generator == "mol2mol":
                            prior = args.m2m_prior or m2m_prior_for(sim_thr)
                            cfg["parameters"]["prior_file"] = prior
                            cfg["parameters"]["agent_file"] = prior
                        cfg["device"] = round_robin_device(device_counter, n_gpus)
                        device_counter += 1
                        cfg["tb_logdir"] = os.path.join(seed_dir, "tb")
                        cfg["parameters"]["summary_csv_prefix"] = os.path.join(seed_dir, "summary")
                        cfg["parameters"]["smiles_file"] = smiles_path
                        cfg["stage"][-1]["chkpt_file"] = os.path.join(seed_dir, "agent.chkpt")

                        params = pmodock_params(cfg)
                        params["target"] = oracle
                        params["oracle_url"] = oracle_url
                        params["history_path"] = os.path.join(seed_dir, "history.csv")
                        params["vina_seed"] = seed
                        # The reward always measures similarity to the lead itself, never to
                        # the scaffold handed to LibInvent.
                        params["lead_smiles"] = lead
                        params["sim_thr"] = sim_thr
                        if not args.hparam_config:
                            params["reward_type"] = args.reward_type
                        set_budget(cfg, args.max_oracle_calls)

                        jobs.append((write_config(cfg, os.path.join(seed_dir, "config.json")), seed))

    logging.info("Running %d jobs with up to %d parallel workers", len(jobs), args.max_workers)
    failed = run_all(jobs, max_workers=args.max_workers)
    logging.info("Completed %d jobs, %d failed", len(jobs), failed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
