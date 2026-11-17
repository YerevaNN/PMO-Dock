#!/bin/bash
# Differential test on CPU: HFBackend vs the pre-port inline generation loop.
#
# Same test as run_hf_differential.slurm. The property -- two implementations,
# one RNG state, identical output -- is device-independent, and research has a
# day-long GPU queue, so this is the one that gives an answer today. The GPU
# job remains the confirmation on the device production uses.
#
#   sbatch genetic_chemllama/slurm/run_hf_differential_cpu.slurm
#
#SBATCH --job-name=gc_hf_diff_cpu
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/gc_tests/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/gc_tests/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export DIFF_TEST_DEVICE=cpu

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/genetic_chemllama/slurm/_test_env.sh"

cd "${PROJECT_ROOT}/genetic_chemllama"
"${TEST_PYTHON}" -m pytest tests/test_hf_backend_differential.py -v
