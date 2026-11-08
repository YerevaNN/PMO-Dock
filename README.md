# PMO-Dock

## What is PMO-Dock?

PMO-Dock is a benchmark for **protein-aware molecular optimization**: given a biological target, search for small molecules that bind well, look drug-like, and are easy to synthesize.

The repo is a **monorepo**. The installable **`benchmark`** package holds shared oracles (QED, SA, docking, similarity), task definitions, and experiment helpers. Four generative methods (`genetic_chemllama`, `genmol`, `saturn`, `genetic_gfn`) live alongside it as runnable research code, each with its own conda environment and `*_runner.py` entrypoints.

## Install

```bash
git clone <this-repo>
cd PMO-Dock
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
pip install -e '.[benchmark-core]'
```

**Per-algorithm conda env** (create only what you need):

```bash
conda env create -f genetic_chemllama/env/environment.yml   # env name: cheml
bash genmol/env/setup.sh                                      # genmol
bash saturn/setup.sh                                         # saturn
# genetic_gfn: see genetic_gfn/multi_objective/README.md
```

**Assets not in git:** `genmol/model.ckpt`, Saturn Mamba prior, GFN `Prior.ckpt` — set paths via env vars below.

RDKit and **Open Babel 3.1.1** are required for docking. Pin this version in every env that runs the docking oracle (inline or HTTP): different Open Babel builds can change 3D conformers and therefore QuickVina scores. Method env files already pin it (e.g. `genetic_chemllama/env/environment.yml`: `openbabel=3.1.1`); the docking service (`./benchmark/docking_oracle/start_oracle.sh`) uses `ORACLE_CONDA_ENV` (default `cheml`), which must provide the same pin.

## Structure and usage

| Path | Role |
|------|------|
| **`benchmark/`** | Shipped library: property computers, docking, tasks, `actives.csv`, `experiment_utils` |
| **`genetic_chemllama/`** | ChemLlama + genetic pool optimization |
| **`genmol/`** | Discrete diffusion (GenMol) |
| **`saturn/`** | Mamba/RNN RL with memory |
| **`genetic_gfn/`** | GA + GFlowNet multi-objective search |

Run `*_runner.py` from the repo root. Results go under `$OUT_DIR/<method>/...` (default `$PROJECT_ROOT/results`).

**Using the library:**

```python
from benchmark.computers import QED, SA, SIMILARITY, select_prop_computer
from benchmark.docking_oracle import DockingOracle
from benchmark.paths import get_project_root, resolve_from_project_root
from benchmark.experiment_utils import get_log_dir, get_job_dir
```

**Docking oracle — two modes**

| Mode | When | How |
|------|------|-----|
| **Inline** (default) | Single run | Import `benchmark.docking_oracle.DockingOracle` in-process. |
| **HTTP service** | Many parallel jobs (e.g. hparam sweep) | Shared QuickVina server; set `DOCKING_VINA_URL` and pass `--vina_url` / `--oracle_url` to runners. |

```bash
export DOCKING_VINA_URL=http://127.0.0.1:5050
./benchmark/docking_oracle/start_oracle.sh
```

For a single experiment, inline docking is usually enough. Use the HTTP service when several processes dock at once.

**Environment variables**

| Variable | Meaning |
|----------|---------|
| `PROJECT_ROOT` | Repo root for configs and assets |
| `OUT_DIR` | Where experiment logs are written (default: `$PROJECT_ROOT/results`) |
| `PYTHONPATH` | Should include `$PROJECT_ROOT` when running algorithm code |
| `DOCKING_VINA_URL` | QuickVina HTTP service URL; unset → inline docking |
| `ORACLE_HOST`, `ORACLE_PORT` | Bind address for `start_oracle.sh` (default `127.0.0.1:5050`) |
| `ORACLE_CONDA_ENV` | Conda env for the docking service (default `cheml`) |
| `SATURN_PRIOR_PATH` | Path to Saturn Mamba checkpoint |
| `GENMOL_MODEL_PATH` | Path to GenMol `model.ckpt` |
| `GENMOL_ROOT` | GenMol tree root (default `$PROJECT_ROOT/genmol`) |

## Tasks

**Hit** — Design new molecules from scratch that fit a protein pocket and pass basic drug-likeness filters.

**Lead** — Start from a known active and evolve it: stay similar to the seed while improving properties; score the best binders among molecules that still look like viable leads.

**Specificity (spec)** — Bind strongly to one target (6nzp) but weakly to related off-targets — selective ligands, not promiscuous binders.

**Novel hit** — Same five proteins as hit. The prompt is QED and SA only. The search reward is the stored docking score divided by 20, times a 1-or-Gaussian term on docking, QED, and SA (`--reward_type gated`, tasks `novel.parp1` … `novel.jak2`).

<table>
  <thead>
    <tr>
      <th align="left">Task</th>
      <th align="left" width="36%">Properties</th>
      <th align="left">Oracle budget</th>
      <th align="left">Seeds</th>
      <th align="left" width="36%">Objective</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>hit</strong></td>
      <td>QED ∈ <strong>[0.5, 1]</strong>, SA ∈ <strong>[1, 5]</strong><br>Docking ≥ target-specific floor</td>
      <td>3000</td>
      <td>0–9</td>
      <td><strong>Hit rate</strong> — share of oracle calls that yield molecules meeting all property cutoffs</td>
    </tr>
    <tr>
      <td><strong>lead</strong></td>
      <td>Similarity ≥ <strong>0.6</strong><br>QED ∈ <strong>[0.6, 1]</strong>, SA ∈ <strong>[1, 4]</strong><br>Docking ≥ target-specific floor</td>
      <td>1000</td>
      <td>0–2</td>
      <td><strong>Best docking score</strong> among molecules that satisfy lead constraints (<code>lead.&lt;protein&gt;_06_0</code> … <code>_06_2</code>; see <code>benchmark/tasks.py</code>)</td>
    </tr>
    <tr>
      <td><strong>spec</strong></td>
      <td>QED ∈ <strong>[0.4, 1]</strong>, SA ∈ <strong>[1, 4]</strong><br>Docking ≥ target-specific floor</td>
      <td>3000</td>
      <td>0–2</td>
      <td><strong>Mean top-10 margin</strong> — average of the ten largest (target docking − antitarget docking) gaps among constraint-passing molecules (<code>spec.6nzp_7uyt</code>, …); see <code>benchmark/metrics/task_metrics.py</code> (<code>k=10</code>)</td>
    </tr>
    <tr>
      <td><strong>novel</strong></td>
      <td>QED, SA, docking ≥ target-specific floor</td>
      <td>3000</td>
      <td>0–4</td>
      <td><strong>Gaussian docking reward</strong> — <code>(DS/20)</code> times a 1-or-Gaussian term on docking, QED, and SA (<code>novel.parp1</code> … <code>novel.jak2</code>, <code>--reward_type gated</code>)</td>
    </tr>
  </tbody>
</table>

Constraint details: `benchmark/tasks.py`, `benchmark/spec_tasks.py`. Metrics: `benchmark/metrics/task_metrics.py`.

## Algorithms

| Method | Runner(s) | Conda env |
|--------|-----------|-----------|
| **genetic_chemllama** | `genetic_chemllama/genetic_runner.py` | `cheml` |
| **genmol** | `genmol/genmol_hit_runner.py`, `genmol_lead_runner.py` | `genmol` |
| **saturn** | `saturn/saturn_hit_runner.py`, `saturn_lead_runner.py` | `saturn` |
| **genetic_gfn** | `genetic_gfn/multi_objective/gen_gfn_hit_runner.py`, `gen_gfn_lead_runner.py` | `genetic_gfn` |

| Method | Idea |
|--------|------|
| **genetic_chemllama** | ChemLlama proposes SMILES; a genetic pool keeps diverse high-scoring molecules; oracle scores QED/SA/docking each round |
| **genmol** | Discrete diffusion model samples and mutates molecules; fragment vocabulary for hit/lead |
| **saturn** | Prior/agent Mamba with RL, experience replay, and optional memory; JSON configs per task |
| **genetic_gfn** | Genetic algorithm + GFlowNet over molecular graphs; multi-objective vector (docking, QED, SA, …) |

**Category-best hyperparameters** (source of truth, including `exp-N` → grid mapping): **[`BEST_HPARAMS.md`](BEST_HPARAMS.md)**. Do not infer the freeze from a `best_*.yaml` filename.

**Examples**

Single run (hit, genetic_chemllama) using the frozen values from that table:

```bash
export PROJECT_ROOT=$PWD PYTHONPATH=$PWD
conda activate cheml
python genetic_chemllama/genetic_runner.py \
  --config_file genetic_chemllama/genetic/configs/best_hit.yaml \
  --task_name hit.parp1 --reward_type hit \
  --seeds 0 1 2 3 4 5 6 7 8 9 --max_oracle_calls 3000
```

Convenience copies of the table (hit/spec budget **3000**, lead **1000**): ChemLlama `genetic/configs/best_{hit,lead,spec}.yaml`, GenMol `genmol_*_best.yaml`, GFN `hparams_best_{hit,lead,spec}.yaml`, Saturn `*_best.json`. If a copy disagrees with `BEST_HPARAMS.md`, the table wins. Novel hit is a separate protocol, not a row of that sweep: `genetic/configs/best_novel.yaml` (`pool_size=20`, `num_prompts=50`, `num_similars=2`, `reward_type=gated`).

```bash
python genetic_chemllama/genetic_runner.py \
  --config_file genetic_chemllama/genetic/configs/best_novel.yaml \
  --task_name novel.parp1 --reward_type gated \
  --seeds 0 1 2 3 4 --max_oracle_calls 3000
```

Hparam sweep (spec, Saturn) — start the docking service first, then launch parallel jobs over a grid defined in `saturn/spec/hparams.yaml` (`sigma`, memory on/off, reward type, …):

```bash
# terminal 1
export DOCKING_VINA_URL=http://127.0.0.1:5050
./benchmark/docking_oracle/start_oracle.sh

# terminal 2
export PROJECT_ROOT=$PWD PYTHONPATH=$PWD OUT_DIR=$PWD/results
export SATURN_PRIOR_PATH=$PROJECT_ROOT/saturn/experimental_reproduction/checkpoint_models/zinc-250k-mamba-epoch-50.prior
conda activate saturn
CUDA_VISIBLE_DEVICES=0,1 python saturn/saturn_hit_runner.py \
  --config_file saturn/spec/spec_best.json \
  --oracle_name 6nzp_7uyt \
  --seeds 0 1 2 \
  --max_oracle_calls 3000 \
  --hparam_config spec/hparams.yaml \
  --oracle_url "$DOCKING_VINA_URL" \
  --max_workers 4 --n_gpus 2
```

Use `--search_range 0 2` to run only the first few hparam combinations while debugging. Other runners accept `--hparam_config` the same way (flat YAML for GenMol/GFN; nested YAML for Saturn).

Result layout: `$OUT_DIR/<method>/<task>/<date>/exp-N/<target>/seed-K/` (hparam runs append `-hparam` to the experiment folder name).

## Related upstream projects

- **Saturn** — [github.com/schwallergroup/saturn](https://github.com/schwallergroup/saturn)
- **GenMol** — [github.com/NVIDIA-Digital-Bio/genmol](https://github.com/NVIDIA-Digital-Bio/genmol)
- **Genetic GFN** — [github.com/GFNOrg/gflownet](https://github.com/GFNOrg/gflownet)

## License

Apache-2.0 (see `pyproject.toml`).
