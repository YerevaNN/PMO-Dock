#!/bin/bash
#SBATCH --job-name=hit_grid
#SBATCH --output=logs/hit_grid_%A_%a.out
#SBATCH --error=logs/hit_grid_%A_%a.err
#SBATCH --array=0-26%9
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12

set -euo pipefail

# This script runs a plain Slurm array "grid search" over:
# - targets: parp1, fa7, 5ht1b
# - kl_coefficient: [0.001, 0.005, 0.0005]
# - rank_coefficient: [0.01, 0.03, 0.05]
#
# Total jobs: 3 targets * 3 kl * 3 rank = 27 jobs (array indices 0..26).
#
# Each job runs multi_objective in production mode with:
# - max_oracle_calls = 3000
# - n_runs = 10  (production seeds are hard-coded to 0..9 in optimizer.py)

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
SLURM_ROOT="${OUT_DIR}/genetic_gfn/slurm_jobs/multi_objective_hit_grid"

mkdir -p "${SLURM_ROOT}/logs"
mkdir -p "${RESULTS_ROOT}"

# NOTE:
# Slurm does NOT expand env vars in #SBATCH --output/--error.
# To store slurm logs under $OUT_DIR, submit with:
#   sbatch --chdir="$SLURM_ROOT" submit_hit_task_grid_slurm.sh
# so the relative logs/ path resolves inside $SLURM_ROOT.

TARGETS=(parp1 fa7 5ht1b)
KL_VALUES=(0.001 0.005 0.0005)
RANK_VALUES=(0.01 0.03 0.05)

NUM_KL=${#KL_VALUES[@]}
NUM_RANK=${#RANK_VALUES[@]}
NUM_COMBOS=$((NUM_KL * NUM_RANK))   # 9
NUM_TARGETS=${#TARGETS[@]}         # 3

TASK_ID=${SLURM_ARRAY_TASK_ID}
TARGET_IDX=$((TASK_ID / NUM_COMBOS))
COMBO_ID=$((TASK_ID % NUM_COMBOS))
KL_IDX=$((COMBO_ID / NUM_RANK))
RANK_IDX=$((COMBO_ID % NUM_RANK))

if [ "$TARGET_IDX" -ge "$NUM_TARGETS" ]; then
  echo "Error: TARGET_IDX out of range ($TARGET_IDX >= $NUM_TARGETS) for TASK_ID=$TASK_ID"
  exit 1
fi

TARGET="${TARGETS[$TARGET_IDX]}"
KL="${KL_VALUES[$KL_IDX]}"
RANK="${RANK_VALUES[$RANK_IDX]}"

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-unset}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-unset}"
echo "Target: ${TARGET}"
echo "kl_coefficient: ${KL}"
echo "rank_coefficient: ${RANK}"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment (edit if your env name differs)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gflownet-pmo

# Change to multi_objective directory (in repo)
cd "${MULTI_OBJ_DIR}"

# Optional: MPS for better GPU sharing (uncomment if you want it)
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}
# export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_log
# mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
# nvidia-cuda-mps-control -d

# Avoid CPU oversubscription
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

MAX_ORACLE_CALLS=3000
N_RUNS=10
ALPHA_VECTOR="1,1,1"
OBJECTIVES="qed,sa,${TARGET}"

# Output directory for this grid cell
BASE_OUT="${RESULTS_ROOT}/grid/${TARGET}/kl_${KL}/rank_${RANK}"
mkdir -p "$BASE_OUT"

# Create a per-job config yaml that overrides kl_coefficient and rank_coefficient
CONFIG_IN="genetic_gfn/hparams_default.yaml"
CONFIG_OUT="${BASE_OUT}/hparams_kl_${KL}_rank_${RANK}.yaml"

python - <<PY
import yaml

cfg_in = "${CONFIG_IN}"
cfg_out = "${CONFIG_OUT}"
kl = float("${KL}")
rank = float("${RANK}")

with open(cfg_in, "r") as f:
    cfg = yaml.safe_load(f)

cfg["kl_coefficient"] = kl
cfg["rank_coefficient"] = rank

with open(cfg_out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Wrote", cfg_out)
PY

python run.py genetic_gfn \
  --objectives "${OBJECTIVES}" \
  --alpha_vector "${ALPHA_VECTOR}" \
  --max_oracle_calls "${MAX_ORACLE_CALLS}" \
  --task production \
  --n_runs "${N_RUNS}" \
  --freq_log 100 \
  --wandb disabled \
  --run_name "${TARGET}_hit_task_kl${KL}_rank${RANK}" \
  --output_dir "${BASE_OUT}" \
  --config_default "${CONFIG_OUT}"

# Optional: stop MPS (if enabled above)
# echo quit | nvidia-cuda-mps-control
# rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "=========================================="
echo "End time: $(date)"
echo "Outputs in: ${BASE_OUT}"
echo "=========================================="

