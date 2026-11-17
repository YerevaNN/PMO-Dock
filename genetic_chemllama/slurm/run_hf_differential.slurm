#!/bin/bash
# Differential test: HFBackend vs the pre-port inline generation loop.
#
# Needs a GPU and the real ChemLlama checkpoint, so it is kept out of the
# CPU-only unit-test job. This is the one that protects every existing hf run
# from the backend refactor.
#
#   sbatch genetic_chemllama/slurm/run_hf_differential.slurm
#
#SBATCH --job-name=gc_hf_diff
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/gc_tests/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/gc_tests/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/genetic_chemllama/slurm/_test_env.sh"

nvidia-smi -L || true

cd "${PROJECT_ROOT}/genetic_chemllama"
"${TEST_PYTHON}" -m pytest tests/test_hf_backend_differential.py -v
