# PMO-Dock

**PMO-Dock** benchmarks **protein-aware molecular optimization** methods that search for small molecules optimized for:

- **Docking** (QuickVina2 in-process by default, optional HTTP service for parallel runs)
- **Drug-likeness** (QED)
- **Synthetic accessibility** (SA)
- **Lead similarity** (Tanimoto to seed actives), for lead tasks
- **Target vs antitarget docking**, for specificity tasks

The installable Python distribution ships the **`benchmark`** package: property computers (QED, SA, similarity), QuickVina docking integration, small datasets/assets (e.g. lead seeds), and helpers for **tasks / rewards / metrics** aligned with the paper experiments.

The Git repository is a **monorepo**: the PyPI-oriented library lives under `benchmark/`, while algorithm implementations (`saturn/`, `genetic_chemalactica/`, `genmol/`, `genetic_gfn/`, etc.) stay in the tree for reproducibility and are **not** shipped as top-level packages in the wheel.

## Install

**From a clone** (typical for development):

```bash
cd /path/to/PMO-Dock
python -m pip install -e '.[benchmark-core]'
```

That installs the **`pmo-dock`** distribution in editable mode. Only the **`benchmark`** package is registered in your environment (see `pyproject.toml`).

**Extras**

- **`benchmark-core`**: pulls in `numpy` and `requests` for computers and HTTP docking clients. Install it unless you manage those pins yourself.

**Heavy scientific stack**

- **RDKit** and **Open Babel** (e.g. `openbabel` Python bindings) are expected for full local docking and cheminformatics. They are usually easiest to install via **conda**/mamba in the same environment; they are not declared as hard `pip` dependencies here.

## Using the library

```python
from benchmark.computers import QED, SA, SIMILARITY, select_prop_computer
from benchmark.docking_oracle import DockingOracle
from benchmark.paths import get_project_root, resolve_from_project_root
from benchmark.experiment_utils import get_log_dir, get_job_dir
```

- **Computers** live in `benchmark.computers` (see `benchmark/computers/property_computers.py`).
- **Docking** client/service code lives in `benchmark.docking_oracle` (grids and related assets are included as package data where configured in `pyproject.toml`).
- **Paths**: set **`PROJECT_ROOT`** to your checkout (or any root that holds `benchmark/actives.csv` and grids) so resolvers point at the right files; if unset, `benchmark.paths` infers the repo root from the installed package layout (works for a normal clone + editable install).

Bundled data includes, among others, **`benchmark/actives.csv`** for lead-style seeds (see `benchmark/actives_loader.py`).

## Benchmark tasks

| Task | Goal | Typical oracle / target naming |
|------|------|--------------------------------|
| **hit** | De novo molecules meeting QED + SA + strong docking on one receptor | `hit.parp1`, … or short `parp1` (GenMol/Saturn) |
| **lead** | Improve from a **seed** SMILES: similarity + QED + SA; rank by docking on top-k | `lead.parp1_04_0` (protein, sim tier 04/06, seed index 0–2) |
| **specificity (spec)** | Strong **6nzp** docking, weak antitarget (**7uyt**, **5ut5**, **7uyw**, **4l00**, **5khw**), plus QED/SA | `spec.6nzp_7uyt`, or `6nzp_7uyt` (GenMol/Saturn) |

**Constraint definitions** (evaluation): `benchmark/tasks.py`  
**Genetic prompts / computer lists**: `genetic_chemalactica/utils/tasks.py`  
**Lead seeds**: `benchmark/actives.csv` (3 SMILES per target protein)

### Hit targets

`parp1`, `fa7`, `5ht1b`, `braf`, `jak2` — per-target docking floors in `benchmark/tasks.py` (`hit.<target>`).

### Lead layout

- Proteins: same five targets as hit.
- Similarity tiers: **0.4** (`lead.sim_04`) and **0.6** (`lead.sim_06`).
- Three seed indices per target from `benchmark/actives.csv`.

### Specificity

- Target receptor: **6nzp**
- Antitargets: **7uyt**, **5ut5**, **7uyw**, **4l00**, **5khw**
- Docking floor on target ≥ 10.67; antitarget dock in [0, 20]; QED ≥ 0.4; SA ∈ [1, 4]

## Algorithms

| Algorithm | Directory | Runner(s) | Notes |
|-----------|-----------|-----------|--------|
| **genetic_chemalactica** | `genetic_chemalactica/` | `genetic_runner.py` | LLM + genetic pool; `--task_name` like `hit.parp1`, `lead.parp1_04_0`, `spec.6nzp_7uyt`; `--reward_type` for `hit.<target>` (`hit`, `max`, `geam`) |
| **genetic_gfn** | `genetic_gfn/multi_objective/` | `gen_gfn_hit_runner.py`, `gen_gfn_lead_runner.py` | GA + GFlowNet; `run.py genetic_gfn`; spec via `--targets 6nzp` + anti-target subdirs |
| **genmol** | `genmol/` | `genmol_hit_runner.py`, `genmol_lead_runner.py` | Discrete diffusion; needs **`genmol/model.ckpt`** |
| **saturn** | `saturn/` | `saturn_hit_runner.py`, `saturn_lead_runner.py` | Mamba/RNN RL + memory; JSON configs under `hit/`, `lead/` |

## Repository layout vs. install

| Area | Role |
|------|------|
| **`benchmark/`** | **Shipped** as the `pmo-dock` library (`import benchmark...`). |
| **`saturn/`**, **`genetic_chemalactica/`**, **`genmol/`**, **`genetic_gfn/`** | Research / experiment code; run from this repo with the **repo root on `PYTHONPATH`** if imports are not under `benchmark`. |

**GenMol** in this tree is a separate package: from the repo root, `python -m pip install -e genmol/env` if you need the `genmol` import path.

So: **`pip install pmo-dock`** (or `pip install -e '.[benchmark-core]'`) gives you **`benchmark`**, not necessarily `saturn` or `genetic_chemalactica`. For those entrypoints, either work from the repo with:

```bash
export PYTHONPATH=/path/to/PMO-Dock
```

or invoke modules in a way your scheduler already sets up.

### Shared infrastructure

| Path | Role |
|------|------|
| `benchmark/experiment_utils.py` | `get_log_dir`, `get_job_dir` for all `*_runner.py` |
| `benchmark/tasks.py` | Hit/lead constraint definitions |
| `benchmark/actives.csv` | Lead seed molecules |
| `benchmark/actives_loader.py` | Lead seed SMILES from `actives.csv` |
| `benchmark/docking_oracle/` | Docking client, grids, optional Flask oracle app (`start_oracle.sh`) |
| `genetic_chemalactica/utils/tasks.py` | Genetic task prompts, computer lists, validation |

## Environment variables

```bash
source .env_vars   # portable defaults; auto-loads .env_vars_dev when present (gitignored)
```

Create **`.env_vars_dev`** locally for machine-specific paths (Saturn prior, ChemLlaMA checkpoints, etc.). It is listed in `.gitignore` and sourced automatically after `.env_vars`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `PROJECT_ROOT` | repo root | Code and config resolution |
| `OUT_DIR` | `$PROJECT_ROOT/results` | Experiment logs |
| `DOCKING_USE_HTTP_SERVICE` | `0` | Set to `1` to use the shared QuickVina HTTP service (parallel / hparam) |
| `DOCKING_VINA_URL` | *(unset unless HTTP mode)* | HTTP service URL; auto-set when `DOCKING_USE_HTTP_SERVICE=1` |
| `ORACLE_HOST` / `ORACLE_PORT` | `127.0.0.1` / `5050` | Bind address when starting the optional HTTP service |
| `ORACLE_CONDA_ENV` | `mol-grpo` | Conda env to run `benchmark.docking_oracle.oracle_app` |
| `SATURN_PRIOR_PATH` | *(set in `.env_vars_dev`)* | Saturn Mamba checkpoint path |

### Conda environments

| Algorithm | Expected env |
|-----------|--------------|
| **genetic_chemalactica** | `mol-grpo` |
| **genmol** | `genmol` |
| **saturn** | `saturn` |
| **genetic_gfn** | `genetic_gfn` (needs torch-geometric, ray, botorch, gpytorch) |

The matrix script does not activate environments automatically; use your scheduler or wrap calls with the appropriate `conda activate`.

### Docking oracle

**Default (single run):** algorithms import `benchmark.docking_oracle.DockingOracle` in-process. No separate server is required. Grids live under `benchmark/docking_oracle/docking_grids/`.

**Optional HTTP service (parallel / hparam):** when many workers dock at once, a shared Flask service serializes QuickVina calls and avoids cross-process contention:

```bash
export DOCKING_USE_HTTP_SERVICE=1
source .env_vars
./benchmark/docking_oracle/start_oracle.sh
# health: curl "$DOCKING_VINA_URL/health"
```

Pass the URL into runners when needed: `--vina_url` (genetic_chemalactica), `--oracle_url` (genmol, saturn, genetic_gfn). If unset, each process uses local in-process docking.

**genetic_chemalactica** also supports an optional internal property HTTP wrapper (`oracle.use_oracles_app: true` in config, or `ORACLES_APP_DISABLE=1` to force local). Default is in-process (`LocalDynamicOracle`).

### Asset paths

| Asset | Location |
|-------|----------|
| Saturn Mamba prior | `saturn/experimental_reproduction/checkpoint_models/zinc-250k-mamba-epoch-50.prior` |
| GenMol checkpoint | `genmol/model.ckpt` (not vendored; place or symlink from parent repo) |
| GFN Prior | `genetic_gfn/multi_objective/genetic_gfn/data/Prior.ckpt` |

## Running experiments

Use the `*_runner.py` entrypoints under each algorithm directory (see examples below). Tune via runner flags and env: `SEEDS`, `MAX_ORACLE_CALLS`, `MAX_WORKERS`.

### Per-algorithm examples (single hit target)

```bash
# genetic_chemalactica (local docking by default)
python genetic_chemalactica/genetic_runner.py \
  --config_file genetic_chemalactica/genetic/configs/best.yaml --seeds 0 1 2 \
  --task_name hit.parp1 --reward_type hit --max_oracle_calls 3000

# genetic_gfn
python genetic_gfn/multi_objective/gen_gfn_hit_runner.py \
  --config_file genetic_gfn/hparams_best.yaml --seeds 0 1 2 \
  --targets parp1 --max_oracle_calls 3000

# genmol
python genmol/genmol_hit_runner.py \
  --config_file scripts/exps/hit/configs/genmol_hit.yaml --seeds 0 1 2 \
  --oracle_name parp1 --max_oracle_calls 3000

# saturn
python saturn/saturn_hit_runner.py \
  --config_file hit/configs/hit.json --seeds 0 1 2 \
  --oracle_name parp1 --max_oracle_calls 3000 --reward_type geam
```

For parallel hparam sweeps, start the HTTP oracle and append `--vina_url "$DOCKING_VINA_URL"` or `--oracle_url "$DOCKING_VINA_URL"` to the matching runner.

## Result layout

```
$OUT_DIR/<method>/<task_or_model>/<YYYY-MM-DD>/exp-N/<oracle>/seed-K/
```

Submitit job metadata: `$OUT_DIR/job_dirs/<category>/...`

## Dependencies (high level)

| Component | Setup |
|-----------|--------|
| Benchmark library | `pip install -e '.[benchmark-core]'` |
| genetic_chemalactica | `conda env create -f genetic_chemalactica/env/environment.yml` |
| genetic_gfn | torch, rdkit, ray, botorch, torch-geometric (see `genetic_gfn/multi_objective/README.md`) |
| genmol | `bash genmol/env/setup.sh`; checkpoint at `genmol/model.ckpt` |
| saturn | `bash saturn/setup.sh` |
| Docking | In-process `DockingOracle`; optional HTTP service via `DockingVinaClient` |

## Known gaps / manual steps

1. **GenMol checkpoint** — not vendored; place `genmol/model.ckpt` or override `model_path` in generated configs.
2. **Saturn prior** — set `SATURN_PRIOR_PATH` in `.env_vars_dev` (configs use `${SATURN_PRIOR_PATH}`).
3. **Per-algorithm conda envs** — runners do not activate environments; use your scheduler or wrap calls.

## Related repos

This tree was split from **[Even-More-PMO](/auto/home/gor/projects/Even-More-PMO)**; missing utilities were restored from there where applicable.

## License

Apache-2.0 (see `pyproject.toml`).
