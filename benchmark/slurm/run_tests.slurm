#!/bin/bash
# benchmark/ unit tests.
#
# CPU-only and dependency-light (numpy and pytest; no OpenBabel, no qvina, no
# GPU), so they belong on research_cpu and finish in seconds. They pin the
# docking oracle's semaphore sizing, which is silent when wrong -- the node
# oversubscribes and nothing in the output says so.
#
#   sbatch benchmark/slurm/run_tests.slurm
#
#SBATCH --job-name=bm_tests
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/bm_tests/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/bm_tests/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

mkdir -p "/mnt/weka/${USER}/results/slurm_logs/bm_tests"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
# The run env (mol-grpo) carries no pytest and is shared -- not ours to install
# into. These tests need only numpy, which base also has. Anything added here
# that needs a run-env import belongs in a separate job against mol-grpo via
# the venv helper in genetic_chemllama/slurm/_test_env.sh.
conda activate "${TEST_ENV:-base}"

echo "python: $(which python)"
python -VV
cd "${PROJECT_ROOT}"
python -m pytest benchmark/tests/ -v
