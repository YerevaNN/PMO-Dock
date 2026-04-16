from __future__ import print_function

import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
# from tdc import Oracle
from time import time 
from typing import Optional, Tuple

def _count_csv_rows(path: str) -> Optional[int]:
    """
    Return number of data rows (excluding header) in a CSV file.
    Best-effort: returns None if file is missing/unreadable.
    """
    try:
        if not path:
            return None
        if not os.path.exists(path):
            return None
        n = 0
        with open(path, "r") as f:
            # first line is header
            _ = f.readline()
            for _line in f:
                n += 1
        return n
    except Exception:
        return None

def _setup_run_logs(*, output_dir: str, run_name: str, log_dir: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Optional per-run stdout/stderr redirection.

    If log_dir is provided, we create:
      <log_dir>/<run_name>.stdout.log
      <log_dir>/<run_name>.stderr.log

    Returns (stdout_path, stderr_path) when enabled, else (None, None).
    """
    if not log_dir:
        return None, None
    os.makedirs(log_dir, exist_ok=True)
    safe_name = (run_name or "run").replace(os.sep, "_")
    stdout_path = os.path.join(log_dir, f"{safe_name}.stdout.log")
    stderr_path = os.path.join(log_dir, f"{safe_name}.stderr.log")
    # Line-buffered text mode.
    sys.stdout = open(stdout_path, "w", buffering=1)
    sys.stderr = open(stderr_path, "w", buffering=1)
    print(f"[run.py] output_dir={output_dir}")
    print(f"[run.py] run_name={run_name}")
    print(f"[run.py] stdout_log={stdout_path}")
    print(f"[run.py] stderr_log={stderr_path}")
    print()
    sys.stdout.flush()
    return stdout_path, stderr_path

def main():
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('method', default='graph_ga')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=1000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=5)
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"]) ### 
    parser.add_argument("--objectives", type=str, default='gsk3b,jnk3,qed,sa')
    parser.add_argument("--alpha_vector", default='1,1,1,1', type=str)
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_code', action='store_true')
    parser.add_argument('--wandb', type=str, default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument('--run_name', type=str, default="default")
    # Optional logging to files (stdout/stderr redirection).
    # If not set, behavior is unchanged (logs go to terminal / slurm stdout).
    parser.add_argument(
        '--log_dir',
        type=str,
        default="",
        help="If set, redirect this run's stdout/stderr to files under this directory."
    )
    # Docking oracle base URL (for docking objectives like parp1/fa7/5ht1b/braf/jak2).
    # Prefer passing this explicitly instead of environment variables.
    parser.add_argument(
        '--oracle_url',
        type=str,
        default="",
        help="Docking oracle base URL (e.g. 127.0.0.1:5454). Overrides any `oracle_url:` in config_default YAML."
    )
    # Lead molecule for similarity objective (Tanimoto to this SMILES).
    # If provided (non-empty), we automatically append `similarity` to objectives unless already present.
    parser.add_argument('--seed_mol', type=str, default="", help="Lead SMILES for `similarity` objective (Tanimoto).")

    # Anti-target docking objective (selectivity): only meaningful for 6nzp selectivity tasks.
    # If objectives contain `6nzp`, this must be provided and will be appended as an additional objective.
    parser.add_argument(
        '--anti_target',
        type=str,
        default="",
        help="Anti-target docking receptor name (e.g. 7uyt/7uyw/5ut5). "
             "Required if objectives include `6nzp`. When set, it is appended to objectives and "
             "its normalized score is transformed as (1 - x).",
    )
    args = parser.parse_args()
    
    args.objectives = args.objectives.split(',')
    args.alpha_vector = args.alpha_vector.split(',')
    args.alpha_vector = [float(x) for x in args.alpha_vector]

    # Similarity objective wiring:
    # - make lead SMILES available to scorer via args (set a module-global in scorer)
    # - auto-append objective + weight if user passed a seed_mol
    if isinstance(args.seed_mol, str) and args.seed_mol.strip():
        try:
            from oracle.scorer.scorer import set_lead_smiles
            set_lead_smiles(args.seed_mol.strip())
        except Exception:
            # Fallback: older paths may still read env var.
            os.environ["LEAD_SMILES"] = args.seed_mol.strip()
        if "similarity" not in [o.lower() for o in args.objectives]:
            args.objectives.append("similarity")
            # If user didn't provide a matching extra alpha weight, default to 1.0
            if len(args.alpha_vector) < len(args.objectives):
                args.alpha_vector.append(1.0)

    # Anti-target docking wiring:
    # - If objectives include 6nzp, require --anti_target.
    # - If provided, append it as an additional objective (unless already present)
    # - Configure scorer to treat that objective as anti-target (normalized score x -> 1-x).
    objectives_lower = [o.lower() for o in args.objectives]
    has_6nzp = "6nzp" in objectives_lower
    anti_target = (args.anti_target or "").strip()
    if has_6nzp and not anti_target:
        raise ValueError("Objectives include `6nzp` but --anti_target was not provided (e.g. --anti_target 7uyt).")
    if anti_target:
        anti_lower = anti_target.lower()
        if anti_lower not in objectives_lower:
            args.objectives.append(anti_target)
            objectives_lower.append(anti_lower)
            if len(args.alpha_vector) < len(args.objectives):
                args.alpha_vector.append(1.0)
        try:
            from oracle.scorer.scorer import set_anti_target
            set_anti_target(anti_target)
        except Exception:
            # No env-var fallback here; treat as best-effort.
            pass

    os.environ["WANDB_MODE"] = args.wandb

    if not args.log_code:
        os.environ["WANDB_DISABLE_CODE"] = "false"

    args.method = args.method.lower() 

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, args.method)

    sys.path.append(path_main)
    
    print(args.method)
    # Add method name here when adding new ones
    if args.method == "genetic_gfn":
        from genetic_gfn.run import Genetic_GFN_Optimizer as Optimizer
    else:
        raise ValueError("Unrecognized method name.")


    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # If requested, redirect stdout/stderr to files (typically inside output_dir/logs).
    # Note: we do this after output_dir is resolved so users can pass relative paths.
    if isinstance(args.log_dir, str) and args.log_dir.strip():
        log_dir = args.log_dir.strip()
        if not os.path.isabs(log_dir):
            log_dir = os.path.join(args.output_dir, log_dir)
        _setup_run_logs(output_dir=args.output_dir, run_name=args.run_name, log_dir=log_dir)

    if args.pickle_directory is None:
        args.pickle_directory = path_main

    if args.task != "tune":
    
        # for oracle_name in args.oracles:

        print(f'Optimizing oracle function: {args.objectives}, {args.alpha_vector}')

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))
        config_default = config_default or {}

        # Parallelism wiring: GeneticGFN uses `config["num_jobs"]` (joblib.Parallel pool).
        # Allow CLI `--n_jobs` to override the YAML when explicitly provided.
        if isinstance(args.n_jobs, int) and args.n_jobs != -1:
            config_default["num_jobs"] = int(args.n_jobs)

        # Docking oracle URL wiring (used by oracle/scorer/scorer.py for docking targets).
        # Priority: CLI --oracle_url > config_default["oracle_url"].
        oracle_url = ""
        try:
            oracle_url = (args.oracle_url or "").strip()
        except Exception:
            oracle_url = ""
        if not oracle_url:
            try:
                oracle_url = str(config_default.get("oracle_url", "")).strip()
            except Exception:
                oracle_url = ""
        if oracle_url:
            try:
                from oracle.scorer.scorer import set_docking_service_url
                set_docking_service_url(oracle_url)
            except Exception:
                # If scorer isn't importable yet for some reason, fall back to env var for legacy paths.
                # (Still discouraged; main scorer has been updated to prefer explicit configuration.)
                os.environ["DOCKING_VINA_URL"] = oracle_url

        # oracle = Oracle(name = oracle_name)
        oracle = (args.objectives, args.alpha_vector)
        optimizer = Optimizer(args=args)

        if args.task == "simple":
            # optimizer.optimize(oracle=oracle, config=config_default, seed=args.seed) 
            for seed in args.seed:
                kl = config_default.get("kl_coefficient", None)
                rank = config_default.get("rank_coefficient", None)
                seed_mol = (args.seed_mol or "").strip()
                print(f"seed={seed} kl={kl} rnk={rank} s_mol={seed_mol!r} out_dir={args.output_dir}")
                run_t0 = time()
                try:
                    optimizer.optimize(oracle=oracle, config=config_default, seed=seed)
                    status = "success"
                except KeyboardInterrupt:
                    status = "keyboard_interrupt"
                    raise
                except Exception as e:
                    status = f"exception:{type(e).__name__}"
                    raise
                finally:
                    run_dt_s = time() - run_t0

                    # End-of-run summary (best-effort; works for both hit and lead tasks).
                    summary = getattr(optimizer, "last_run_summary", None) or {}
                    stop_reason = summary.get("stop_reason") or ""
                    n_unique = summary.get("n_unique_molecules")
                    results_yaml = summary.get("results_yaml") or ""
                    mol_csv = summary.get("molecules_csv") or ""
                    mol_csv_rows = _count_csv_rows(mol_csv) if mol_csv else None

                    print(
                        "[summary] "
                        f"seed={seed} status={status} "
                        f"stop_reason={stop_reason!r} "
                        f"runtime_s={run_dt_s:.1f} runtime_m={run_dt_s/60.0:.2f} "
                        f"n_unique_molecules={n_unique} "
                        f"results_yaml={results_yaml!r} "
                        f"molecules_csv={mol_csv!r} "
                        f"molecules_csv_rows={mol_csv_rows}"
                    )
        elif args.task == "production":
            run_t0 = time()
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)
            run_dt_s = time() - run_t0
            print(f"production finished in {run_dt_s:.1f}s ({run_dt_s/60.0:.2f}m)")
        else:
            raise ValueError('Unrecognized task name, task should be in one of simple, tune and production.')

    # elif args.task == "tune":

    #     print(f'Tuning hyper-parameters on tasks: {args.oracles}')

    #     try:
    #         config_default = yaml.safe_load(open(args.config_default))
    #     except:
    #         config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

    #     try:
    #         config_tune = yaml.safe_load(open(args.config_tune))
    #     except:
    #         config_tune = yaml.safe_load(open(os.path.join(path_main, args.config_tune)))

    #     oracles = [Oracle(name = oracle_name) for oracle_name in args.oracles]
    #     optimizer = Optimizer(args=args)
        
    #     optimizer.hparam_tune(oracles=oracles, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)

    else:
        raise ValueError('Unrecognized task name, task should be in one of simple, tune and production.')
    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))
    # print('If the program does not exit, press control+c.')


if __name__ == "__main__":
    main()

