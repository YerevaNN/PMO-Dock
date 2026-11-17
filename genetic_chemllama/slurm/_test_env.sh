#!/bin/bash
# Shared helper: give the run env a pytest without modifying it.
#
# The tests that need torch/transformers must run in the run env (mol-grpo),
# which carries no pytest and is shared -- not ours to install into. base has
# pytest but a different Python (3.13 vs 3.10), so borrowing it across envs
# fails at import. So: a venv layered on the run env with --system-site-packages,
# which sees all of mol-grpo's packages and adds pytest on top.
#
# Lives under /mnt/weka/$USER per the repo's "no large files outside weka" rule.
# Created on first use and reused afterwards.
#
# Usage:  source genetic_chemllama/slurm/_test_env.sh   (sets TEST_PYTHON)

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/mnt/weka/shared-cache/miniforge3}"
CHEM_ENV="${CHEM_ENV:-mol-grpo}"
TEST_VENV="${TEST_VENV:-/mnt/weka/${USER}/envs/chemllama-test}"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CHEM_ENV}"

if [[ ! -x "${TEST_VENV}/bin/python" ]]; then
  echo "TEST_ENV: creating ${TEST_VENV} (layered on ${CHEM_ENV})"
  python -m venv --system-site-packages "${TEST_VENV}"
  "${TEST_VENV}/bin/python" -m pip install --quiet --upgrade pip
  "${TEST_VENV}/bin/python" -m pip install --quiet pytest
fi

if ! "${TEST_VENV}/bin/python" -c "import pytest" 2>/dev/null; then
  "${TEST_VENV}/bin/python" -m pip install --quiet pytest
fi

export TEST_PYTHON="${TEST_VENV}/bin/python"
echo "TEST_ENV: TEST_PYTHON=${TEST_PYTHON}"
"${TEST_PYTHON}" -c "import sys, torch, pytest; print('python', sys.version.split()[0], 'torch', torch.__version__, 'pytest', pytest.__version__)"
