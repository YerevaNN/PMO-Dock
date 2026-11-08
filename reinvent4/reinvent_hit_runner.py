#!/usr/bin/env python
"""Run REINVENT4 on PMO-Dock hit and spec tasks.

Hit and spec share one runner because both are de novo: the Reinvent SMILES-RNN prior
with no seed structure. ``--oracle_name`` selects the task -- ``parp1`` for hit,
``6nzp_7uyt`` for spec -- and the PMODock component derives the constraints.

Example (hit, single config, 10 seeds):

    export PROJECT_ROOT=$PWD PYTHONPATH=$PWD OUT_DIR=$PWD/results
    export DOCKING_VINA_URL=http://127.0.0.1:5050
    export REINVENT_PRIOR_BASE=$PROJECT_ROOT/reinvent4/priors
    conda activate reinvent4
    python reinvent4/reinvent_hit_runner.py \
      --config_file reinvent4/configs/hit.json \
      --oracle_name parp1 --seeds 0 1 2 3 4 5 6 7 8 9 \
      --max_oracle_calls 3000 --reward_type geam
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--oracle_name", nargs="+", required=True, type=str)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--max_oracle_calls", required=True, type=int)
    parser.add_argument("--reward_type", type=str, default="geam", choices=["geam", "hit"])
    parser.add_argument("--hparam_config", type=str, default=None)
    parser.add_argument(
        "--oracle_url",
        type=str,
        default=None,
        help="Docking HTTP service URL. Required (here or as DOCKING_VINA_URL): the "
        "reinvent4 env has no OpenBabel/QuickVina for inline docking.",
    )
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

    get_job_dir(args.hparam_config is not None, cat="reinvent4-hit")
    n_gpus = infer_n_gpus(args.n_gpus)
    logging.info(
        "device assignment: round_robin (n_gpus=%s, CUDA_VISIBLE_DEVICES=%s)",
        n_gpus,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

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

        for oracle in args.oracle_name:
            oracle_dir = os.path.join(log_dir, oracle)
            os.makedirs(oracle_dir, exist_ok=True)

            for seed in args.seeds:
                seed_dir = os.path.join(oracle_dir, f"seed-{seed}")
                os.makedirs(seed_dir, exist_ok=True)

                cfg = copy.deepcopy(config)
                cfg["device"] = round_robin_device(device_counter, n_gpus)
                device_counter += 1
                cfg["tb_logdir"] = os.path.join(seed_dir, "tb")
                cfg["parameters"]["summary_csv_prefix"] = os.path.join(seed_dir, "summary")
                cfg["stage"][-1]["chkpt_file"] = os.path.join(seed_dir, "agent.chkpt")

                params = pmodock_params(cfg)
                params["target"] = oracle
                params["oracle_url"] = oracle_url
                params["history_path"] = os.path.join(seed_dir, "history.csv")
                params["vina_seed"] = seed
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
