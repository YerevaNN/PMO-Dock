#!/usr/bin/env bash
# Set up the reinvent4 environment and fetch the priors.
#
# Run this through SLURM, not on the login node:
#     sbatch reinvent4/slurm/setup.slurm
# The conda solver needs several GB of RAM and login-node shells here are cgroup-capped
# at 4 GiB, where it gets OOM-killed during "Collecting package metadata".
#
# Everything large lives under /mnt/weka/$USER:
#   checkout + priors  ->  $WEKA_ROOT/large-assets/reinvent4/src   (~1.5 GB)
#   conda env          ->  $WEKA_ROOT/envs/reinvent4               (~8 GB with torch)
#
# REINVENT4 is pinned to tag v4.5.11 for two reasons:
#   1. v4.5.11 still ships every prior in-repo (reinvent, libinvent, linkinvent and all
#      five mol2mol flavours). Later versions moved priors to Zenodo, where the only
#      published record has neither reinvent.prior nor any mol2mol_*.prior.
#   2. Its pins (torch==2.5.1+cu124, numpy<2, rdkit>=2021, python>=3.10) coexist with the
#      benchmark package. HEAD pins torch==2.12.0, numpy>=2 and pandas>=3.
#
# Docking runs through the HTTP oracle service, so OpenBabel and QuickVina are NOT
# installed here -- they stay in the vina-oracle / cheml env behind start_oracle.sh.
set -euo pipefail

REINVENT_TAG="${REINVENT_TAG:-v4.5.11}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WEKA_ROOT="${WEKA_ROOT:-/mnt/weka/$USER}"
SRC_DIR="${REINVENT4_SRC:-$WEKA_ROOT/large-assets/reinvent4/src}"
ENV_PREFIX="${REINVENT4_ENV:-$WEKA_ROOT/envs/reinvent4}"
# The checkout already holds the priors; no copy, no symlink farm.
PRIOR_DIR="${REINVENT_PRIOR_BASE:-$SRC_DIR/priors}"

echo "==> project root : $PROJECT_ROOT"
echo "==> reinvent src : $SRC_DIR ($REINVENT_TAG)"
echo "==> conda env    : $ENV_PREFIX"
echo "==> priors       : $PRIOR_DIR"

mkdir -p "$(dirname "$SRC_DIR")" "$(dirname "$ENV_PREFIX")"

if [ ! -d "$SRC_DIR/.git" ]; then
  git clone --depth 1 --branch "$REINVENT_TAG" \
    https://github.com/MolecularAI/REINVENT4.git "$SRC_DIR"
else
  echo "==> source already present, skipping clone"
fi

missing=0
for prior in reinvent libinvent linkinvent \
             mol2mol_high_similarity mol2mol_medium_similarity \
             mol2mol_mmp mol2mol_scaffold mol2mol_scaffold_generic; do
  if [ ! -f "$PRIOR_DIR/$prior.prior" ]; then
    echo "!!  missing prior: $PRIOR_DIR/$prior.prior"
    missing=1
  fi
done
[ "$missing" -eq 0 ] || { echo "Priors incomplete; is the checkout at $REINVENT_TAG?"; exit 1; }
echo "==> all 8 required priors present"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
else
  echo "==> env already exists, skipping create"
fi

conda activate "$ENV_PREFIX"
echo "==> python: $(command -v python) ($(python -V 2>&1))"

# rdkit's pip wheel links against libXrender/libXext, which reinvent.runmodes pulls in via
# rdkit.Chem.Draw (runmodes/utils/image.py). Some nodes have them system-wide and some do
# not -- the login node does not, gpu03 does -- so install them into the env rather than
# depending on which node a job lands on.
echo "==> installing X11 libraries rdkit.Chem.Draw links against"
conda install -y -p "$ENV_PREFIX" -c conda-forge xorg-libxrender xorg-libxext

# Install REINVENT4's dependencies without its hard openEye-toolkits requirement, which
# needs a commercial licence and is only used by optional scoring components we never
# score with. chemprop/descriptastorus are dropped for the same reason: they are only
# reachable through scoring components this integration does not use, and chemprop 1.5
# constrains torch away from the 2.5.1 pin.
REQ_FILE="$(mktemp -t reinvent4-requirements.XXXXXX.txt)"
trap 'rm -f "$REQ_FILE"' EXIT

python - "$SRC_DIR/pyproject.toml" <<'PY' > "$REQ_FILE"
import re, sys

SKIP = ("openeye", "chemprop", "descriptastorus")
text = open(sys.argv[1]).read()
block = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M).group(1)
for line in block.splitlines():
    dep = line.split("#")[0].strip().strip(",").strip('"').strip()
    if not dep:
        continue
    if dep.lower().startswith(SKIP):
        print(f"# skipped: {dep}", file=sys.stderr)
        continue
    print(dep)
PY

echo "==> installing REINVENT4 dependencies"
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r "$REQ_FILE"

# v4.5.11's [project].dependencies is incomplete. Both of these are imported at module
# scope somewhere under reinvent.runmodes, so `import reinvent.runmodes` -- and therefore
# the whole CLI -- fails without them:
#   torchvision  reinvent/runmodes/utils/image.py:10   (version must match torch 2.5.1)
#   scipy        reinvent/runmodes/utils/plot.py:3
#
# Four other third-party imports are missing from the env but safely unreachable, so they
# stay uninstalled: caffe2 (function-local, behind a runtime check), reinvent_scoring
# (TYPE_CHECKING only), torchinfo (try/except ImportError), and mols2grid/seaborn
# (reinvent/notebooks/ only). polars is needed only by the separate reinvent_datapre
# data-pipeline entry point, which this integration does not use.
echo "==> installing undeclared upstream dependencies (torchvision, scipy)"
pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  "torchvision==0.20.1+cu124" "scipy"

echo "==> installing REINVENT4 itself (no deps; handled above)"
pip install --no-deps -e "$SRC_DIR"

echo "==> installing the benchmark package"
pip install -e "$PROJECT_ROOT"

echo "==> verifying"
PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/reinvent4/plugins:${PYTHONPATH:-}" \
  python -c "
import torch, rdkit, numpy
import reinvent
from reinvent_plugins.components import comp_pmodock
print('torch   ', torch.__version__)
print('rdkit   ', rdkit.__version__)
print('numpy   ', numpy.__version__)
print('reinvent', reinvent.version.__version__)
print('plugin  ', comp_pmodock.PMODock.__name__, 'discoverable')
"

cat <<EOF

==> done. Add to your environment before running:

  export PROJECT_ROOT=$PROJECT_ROOT
  export PYTHONPATH=\$PROJECT_ROOT:\$PROJECT_ROOT/reinvent4/plugins:\$PYTHONPATH
  export REINVENT_PRIOR_BASE=$PRIOR_DIR
  export DOCKING_VINA_URL=http://<oracle-host>:5050
  conda activate $ENV_PREFIX
EOF
