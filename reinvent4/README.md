# REINVENT4 on PMO-Dock

[REINVENT4](https://github.com/MolecularAI/REINVENT4) (AstraZeneca, Apache 2.0) is the
reference RL framework for generative molecular design. It enters this benchmark for two
reasons:

1. **Seeded generators.** Mol2Mol and LibInvent take a starting molecule as *input*. On
   the lead task — stay within a Tanimoto radius of a known active while docking better —
   the constraint is carried by the model, not only by the reward. No other method here
   can do that; `saturn`, `genmol`, `genetic_gfn` and `genetic_chemllama` are all de
   novo or mutation-based.
2. **A clean ablation against `saturn`.** Both optimise the same DAP augmented-likelihood
   objective (`saturn` is built on REINVENT 3.2). REINVENT4 takes **one** gradient step
   per oracle batch; `saturn` takes N augmentation rounds with selective-purge replay. At
   matched budget, the gap between them measures what Augmented Memory buys.

Be explicit about this in write-ups: REINVENT4 is not a stronger optimiser than `saturn`.
On the RL objective `saturn` is a strict superset. REINVENT4 is the unmeasured industrial
baseline and the only seeded generator in the set.

## Version pin

**`v4.5.11`**, not HEAD. Two reasons:

- HEAD pins `torch==2.12.0` (exact), `numpy>=2`, `pandas>=3`, `rdkit>=2025.09.1`.
  `v4.5.11` pins `torch==2.5.1+cu124`, `numpy<2`, `rdkit>=2021`, python ≥3.10 — which
  coexists with the `benchmark` package.
- HEAD moved the priors to [Zenodo](https://doi.org/10.5281/zenodo.15641296), whose only
  published version has **neither `reinvent.prior` nor any `mol2mol_*.prior`**. `v4.5.11`
  still carries all of them in-repo as plain git blobs.

## Install

Everything runs through SLURM. Login shells here are cgroup-capped at 4 GiB, which is
enough to OOM-kill the conda solver.

```bash
sbatch reinvent4/slurm/setup.slurm
```

This clones `v4.5.11` to `/mnt/weka/$USER/large-assets/reinvent4/src` (code **and**
priors, ~1.5 GB), creates the conda env at `/mnt/weka/$USER/envs/reinvent4` (python 3.11),
installs REINVENT4's dependencies minus three, adds `torchvision` (undeclared upstream),
installs the `benchmark` package, and verifies that the plugin is discoverable.

Dropped dependencies, all unreachable from any component we score with:
`openEye-toolkits` (commercial licence), `chemprop` (also constrains torch away from the
2.5.1 pin), `descriptastorus` (chemprop's).

The job is idempotent — safe to resubmit after a failure.

Then, for every run (the `slurm/*.slurm` scripts do this for you):

```bash
export PROJECT_ROOT=/home/gsimonyan/PMO-Dock
export WEKA_ROOT=/mnt/weka/$USER
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/reinvent4/plugins:$PYTHONPATH
export REINVENT_PRIOR_BASE=$WEKA_ROOT/large-assets/reinvent4/src/priors
export OUT_DIR=$WEKA_ROOT/results
export DOCKING_VINA_URL=http://<oracle-host>:5050
conda activate $WEKA_ROOT/envs/reinvent4
```

**Docking must run through the HTTP service.** The `reinvent4` env has no OpenBabel or
QuickVina; those stay behind `./benchmark/docking_oracle/start_oracle.sh`. Both runners
refuse to start without `--oracle_url` or `DOCKING_VINA_URL`.

## How the bridge works

`plugins/reinvent_plugins/components/comp_pmodock.py` is a REINVENT4 scoring component
discovered through its native namespace-package plugin mechanism — no upstream code is
patched. It is a *native namespace package*, so **do not add `__init__.py`** anywhere
under `reinvent4/plugins/`; that breaks discovery.

It exposes one endpoint (the aggregated benchmark reward in [0,1]) and reuses the shared
oracles directly: `DockingOracleClient`, `benchmark.rewards.hit_reward`,
`task_name2constraints`, `sascorer`. Both reward shapings are supported, matching
`saturn/oracles/docking/geam_oracle.py`:

- `--reward_type geam` — `clip(dock,0,20)/20 × QED × (10−SA)/9`, times similarity on lead,
  times the normalised margin on spec.
- `--reward_type hit` — product of Gaussian distance-to-constraint-window terms.

### Oracle budget

REINVENT4 has **no budget concept**: its only stop condition is `max_steps`, and its
per-component cache silently deduplicates — so `batch_size × steps` badly overstates the
real call count. The component owns the budget instead, with exactly `saturn`'s semantics
(`saturn/oracles/oracle.py:148`):

- one oracle call == one **distinct canonical** SMILES actually scored;
- invalid, in-batch-duplicate and filter-rejected molecules cost nothing (REINVENT4 masks
  them before scoring);
- repeats across steps are served from the component's own cache.

`no_cache = True` disables REINVENT4's cache so the component does its own accounting.
That matters because REINVENT4's cache keys on the raw (possibly randomized) SMILES
rather than the canonical form, and because it is rebuilt at every `[[stage]]` boundary,
which would re-score molecules in a curriculum run.

When the budget is spent the history is flushed and `BudgetExhausted` is raised. It
subclasses REINVENT4's `StageInterruptedControlled`, so the run ends through REINVENT4's
graceful path: checkpoint written, summaries flushed, exit 0. `max_steps` is set to
`ceil(budget / batch_size) × 20` purely as a safety net.

### Output

`history.csv` per run, with the same columns `saturn` writes, so
`benchmark/metrics/task_metrics.py` works unchanged:

```
oracle_calls, smiles, scaffold, reward, raw_vina, qed, raw_sa, aggregated_reward
          [+ raw_vina_antitarget on spec]  [+ raw_similarity on lead]
```

Layout follows `get_log_dir()`:
`$OUT_DIR/reinvent4/<config>/<date>/exp-N/<target>/seed-K/` for hit and spec, and
`.../exp-N/start_mol_idx-J/<target>/seed-K/` for lead.

## Running

All runs go through SLURM. Each script takes the docking oracle URL as its first
argument and forwards any further arguments to the runner, so the defaults can be
overridden without editing the file.

Start the docking service first — REINVENT4 has no inline docking:

```bash
export DOCKING_VINA_URL=http://127.0.0.1:5050
./benchmark/docking_oracle/start_oracle.sh
```

### hit (budget 3000, 5 proteins × seeds 0–9)

```bash
sbatch reinvent4/slurm/hit.slurm http://<oracle-host>:5050
```

### spec (budget 3000, 5 antitarget pairs × seeds 0–2)

```bash
sbatch reinvent4/slurm/spec.slurm http://<oracle-host>:5050
```

### lead (budget 1000) — the differentiating run

```bash
# Mol2Mol, similarity tier 0.6 (prior .m2m_high; tier 04 uses .m2m_medium)
sbatch reinvent4/slurm/lead.slurm http://<oracle-host>:5050

# tier 0.4
sbatch reinvent4/slurm/lead.slurm http://<oracle-host>:5050 --sim_thresholds 0.4

# LibInvent, R-group decoration on the seed's Murcko scaffold
sbatch reinvent4/slurm/lead.slurm http://<oracle-host>:5050 \
  --generator libinvent --config_file reinvent4/configs/lead_libinvent.json
```

### hparam sweeps

```bash
sbatch reinvent4/slurm/hit.slurm http://<oracle-host>:5050 \
  --hparam_config hit/hparams.yaml --oracle_name parp1 --seeds 0 1 2 \
  --search_range 0 4            # first 4 combinations while debugging
```

`hit/hparams.yaml` and `spec/hparams.yaml` give 12 combinations (σ × batch_size × DF);
`lead/hparams.yaml` gives 8 (σ × batch_size × distance_threshold).

### Smoke test before any sweep

```bash
sbatch reinvent4/slurm/hit.slurm http://<oracle-host>:5050 \
  --oracle_name parp1 --seeds 0 --max_oracle_calls 300
```

Then check that `history.csv` has exactly 300 data rows. That is the budget invariant
against the real sampling loop, which the unit tests cannot cover.

## Tests

```bash
sbatch reinvent4/slurm/tests.slurm
```

Run on a compute node, not the login node: `rdkit.Chem.Draw` (reached via
`reinvent.runmodes`) links against X11 libraries, and `reinvent.runmodes.RL.validation`
is needed for the config checks.

Two suites:

- **`test_budget_accounting.py`** — pins the budget semantics, the one place a silent bug
  would invalidate every comparison in the paper.
- **`test_configs_validate.py`** — validates every shipped config against upstream's
  pydantic `RLConfig`, asserts PMODock is the only scoring component (so REINVENT4's
  aggregator stays a no-op), checks that seeded configs declare a `smiles_file` and de
  novo ones do not, that inception appears only where upstream supports it, and that
  every prior the configs name resolves on disk. Skipped when REINVENT4 is absent.

The budget suite alone also runs anywhere with rdkit:

```bash
PYTHONPATH=$PWD:$PWD/reinvent4/plugins pytest reinvent4/tests/test_budget_accounting.py -q
```

These pin the budget semantics — the one place where a silent bug would invalidate every
comparison in the paper. They run without REINVENT4 installed: `tests/conftest.py` stands
in equivalents for the two upstream symbols the component imports (`add_tag`,
`ComponentResults`) when `reinvent` is not importable, and steps aside when it is.

## Known limitations

- **v4.5.11's dependency list is incomplete.** `torchvision` and `scipy` are imported at
  module scope under `reinvent.runmodes` but absent from `[project].dependencies`, so the
  CLI cannot start without them. `setup.sh` installs both explicitly. Four more
  third-party imports are missing from the env but safely unreachable (`caffe2`,
  `reinvent_scoring`, `torchinfo`, and `mols2grid`/`seaborn` in notebooks only); `polars`
  is needed only by the `reinvent_datapre` entry point we do not use.
- **`benchmark.docking_oracle.docking` imports OpenBabel lazily** (inside
  `DockingOracle.docking()`) so that its receptor constants and predicates are importable
  in envs without OpenBabel, which includes this one. Do not move that import back to
  module scope.
- **Inception (experience replay) is Reinvent-RNN only.** `setup/inception.py` guards on
  `model_type == "Reinvent"` and carries an explicit `FIXME` for Lib-/LinkInvent. So lead
  runs get no replay while hit and spec do — a real asymmetry against `saturn`, which
  replays everywhere. Report it rather than hiding it.
- **LibInvent covers 14 of the 15 lead seeds.** `5ht1b` start molecule 2 is all scaffold
  and has no R-groups to decorate; the runner raises and points at `--generator mol2mol`.
  Mol2Mol covers all 15, which is why it is the primary lead generator.
- **Lead constraint windows follow `saturn`'s convention**: the per-protein docking, QED
  and SA windows come from `hit.<protein>`, with the similarity window from
  `lead.sim_<tier>`. `saturn`'s `HITOracle` hardcodes `[0.4, 1.0]` for similarity even on
  the 0.6 tier; this component uses the tier's real window, so `--reward_type hit` on tier
  06 is *not* bit-identical to `saturn` there. `--reward_type geam` is unaffected.
- **Seeding is quirky upstream.** `reinvent/Reinvent.py` only calls `set_seed` when
  `--seed` is passed on the command line, but seeds from the *config* value. The runners
  set both; don't invoke the `reinvent` CLI by hand without doing the same.
- **Do not set `parallel` in `[scoring]`.** The component is one HTTP call; parallel
  scoring would move it into subprocesses, where neither the budget counter nor
  `BudgetExhausted` propagates.
