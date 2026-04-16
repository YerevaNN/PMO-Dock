# PMO-Dock

**PMO-Dock** is the installable Python distribution for benchmark-style **evaluators** used in protein-aware molecular optimization: property computers (QED, synthetic accessibility, similarity), **QuickVina** docking integration, small **datasets/assets** (e.g. lead seeds), and helpers for **tasks / rewards / metrics** aligned with the paper experiments.

The Git repository is a **monorepo**: the PyPI-oriented library lives under `benchmark/`, while algorithm implementations (`saturn/`, `genetic_chemalactica/`, `genmol/`, `genetic_gfn/`, etc.) stay in the tree for reproducibility and are **not** shipped as top-level packages in the wheel.

## Install

**From a clone** (typical for development):

```bash
cd /path/to/Even-More-PMO
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
```

- **Computers** live in `benchmark.computers` (see `benchmark/computers/property_computers.py`).
- **Docking** client/service code lives in `benchmark.docking_oracle` (grids and related assets are included as package data where configured in `pyproject.toml`).
- **Paths**: set **`PROJECT_ROOT`** to your checkout (or any root that holds `benchmark/actives.csv` and grids) so resolvers point at the right files; if unset, `benchmark.paths` infers the repo root from the installed package layout (works for a normal clone + editable install).

Bundled data includes, among others, **`benchmark/actives.csv`** for lead-style seeds (see `benchmark/actives_loader.py`).

## Repository layout vs. install

| Area | Role |
|------|------|
| **`benchmark/`** | **Shipped** as the `pmo-dock` library (`import benchmark...`). |
| **`saturn/`**, **`genetic_chemalactica/`**, **`genmol/`**, **`genetic_gfn/`**, **`utils/`** | Research / experiment code; run from this repo with the **repo root on `PYTHONPATH`** if imports are not under `benchmark`. |
| **`docs/`** | Human-oriented notes (e.g. metrics and result layouts). |

**GenMol** in this tree is a separate package: from the repo root, `python -m pip install -e genmol/env` if you need the `genmol` import path.

So: **`pip install pmo-dock`** (or `pip install -e '.[benchmark-core]'`) gives you **`benchmark`**, not necessarily `saturn` or `genetic_chemalactica`. For those entrypoints, either work from the repo with:

```bash
export PYTHONPATH=/path/to/Even-More-PMO
```

or invoke modules in a way your scheduler already sets up.

## Documentation

- **[Molecular optimization metrics and result layout](docs/MOLECULAR_OPTIMIZATION_METRICS.md)** — hit / lead / specificity tasks, CSV conventions, and how metrics are defined.

## License

Apache-2.0 (see `pyproject.toml`).
