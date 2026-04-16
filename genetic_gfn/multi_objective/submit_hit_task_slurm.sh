#!/bin/bash
#SBATCH --job-name=hit_task
#SBATCH --output=logs/hit_task_%A_%a.out
#SBATCH --error=logs/hit_task_%A_%a.err
#SBATCH --array=1-5%4
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12

set -euo pipefail

OUT_DIR="${OUT_DIR:?Please set OUT_DIR to the desired root for results + slurm logs (e.g. export OUT_DIR=/path/to/out)}"

# Repo root:
# - If you export REPO_ROOT, we use it.
# - Otherwise we fall back to locating it from this script's location.
#
# Example:
#   export REPO_ROOT=/path/to/genetic_gfn
SCRIPT_PATH="$(python - <<'PY'
import os, sys
print(os.path.realpath(sys.argv[1]))
PY
"$0")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MULTI_OBJ_DIR="${REPO_ROOT}/multi_objective"

# We want ALL outputs rooted at OUT_DIR.
RESULTS_ROOT="${OUT_DIR}/genetic_gfn/results"
SLURM_ROOT="${OUT_DIR}/genetic_gfn/slurm_jobs/multi_objective_hit_task"

# Create logs directory if it doesn't exist
mkdir -p "${SLURM_ROOT}/logs"
mkdir -p "${RESULTS_ROOT}"

# NOTE:
# Slurm does NOT expand env vars in #SBATCH --output/--error.
# To store slurm logs under $OUT_DIR, submit with:
#   sbatch --chdir="$SLURM_ROOT" submit_hit_task_slurm.sh
# so the relative logs/ path resolves inside $SLURM_ROOT.

# All 5 targets for hit task
TARGETS=(
    'parp1'
    'fa7'
    '5ht1b'
    'braf'
    'jak2'
)

# Get target name for this array task
TARGET_NAME="${TARGETS[$((SLURM_ARRAY_TASK_ID-1))]}"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Target: $TARGET_NAME"
echo "Method: genetic_gfn"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gflownet-pmo

# Change to multi_objective directory (in repo)
cd "${MULTI_OBJ_DIR}"

# ---- MPS for better GPU sharing (A100) ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

# Avoid CPU oversubscription when running 2 python processes
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

# Configuration
OBJECTIVES="qed,sa,${TARGET_NAME}"
ALPHA_VECTOR="1,1,1"
MAX_ORACLE_CALLS=3000
N_RUNS=5
BASE_OUT="${RESULTS_ROOT}/${TARGET_NAME}_hit_task"
mkdir -p "$BASE_OUT"

# Use production mode with n_runs=5
# Production mode uses seeds: [0, 1, 2, 3, 5] for n_runs=5
# Each run will have its seed number in the CSV filename
srun --exclusive -n1 bash -lc "
  python run.py genetic_gfn \
    --objectives ${OBJECTIVES} \
    --alpha_vector ${ALPHA_VECTOR} \
    --max_oracle_calls ${MAX_ORACLE_CALLS} \
    --task production \
    --n_runs ${N_RUNS} \
    --output_dir ${BASE_OUT} \
    --run_name ${TARGET_NAME}_hit_task \
    --wandb disabled \
    --freq_log 100
"

# ---- Stop MPS ----
echo quit | nvidia-cuda-mps-control
rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "=========================================="
echo "End time: $(date)"
echo "Target $TARGET_NAME completed"
echo "CSV files saved in: ${BASE_OUT}/seed_*/genetic_gfn/results/"
echo "=========================================="

