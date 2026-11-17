#!/bin/bash
# genetic_chemllama unit tests.
#
# These are CPU-only and dependency-light (no torch, no GPU, no vLLM server),
# so they belong on research_cpu and finish in seconds. They exist because the
# generation-backend failures they guard are silent: the run gets faster and
# produces nothing.
#
#   sbatch genetic_chemllama/slurm/run_tests.slurm
#
#SBATCH --job-name=gc_tests
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/gc_tests/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/gc_tests/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
# The run env (mol-grpo) carries no pytest, and it is shared -- not ours to
# install into. These tests need only `requests`, which base also has, so they
# run in base. Anything added here that needs a run-env import belongs in a
# separate job against CHEM_ENV instead.
conda activate "${TEST_ENV:-base}"

echo "python: $(which python)"
python -VV
cd "${PROJECT_ROOT}/genetic_chemllama"
python -m pytest tests/ -v
