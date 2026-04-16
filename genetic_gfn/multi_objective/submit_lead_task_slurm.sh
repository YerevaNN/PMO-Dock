#!/bin/bash
#SBATCH --job-name=lead_task
#SBATCH --output=logs/lead_task_%A_%a.out
#SBATCH --error=logs/lead_task_%A_%a.err
#SBATCH --array=1-90%20
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
SLURM_ROOT="${OUT_DIR}/genetic_gfn/slurm_jobs/multi_objective_lead_task"

# Create logs directory if it doesn't exist
mkdir -p "${SLURM_ROOT}/logs"
mkdir -p "${RESULTS_ROOT}"

# NOTE:
# Slurm does NOT expand env vars in #SBATCH --output/--error.
# To store slurm logs under $OUT_DIR, submit with:
#   sbatch --chdir="$SLURM_ROOT" submit_lead_task_slurm.sh
# so the relative logs/ path resolves inside $SLURM_ROOT.

# Configuration: 3 targets × 5 seeds × 3 seed molecules × 2 thresholds = 90 tasks
TARGETS=('parp1' 'fa7' '5ht1b')
SEEDS=(1 2 3 4 5)
SEED_MOL_INDICES=(0 1 2)
THRESHOLDS=(0.4 0.6)

# Calculate indices for this array task (1-indexed)
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))

# Calculate which combination this task is
# Order: iterate through targets, then seeds, then seed_mol_indices, then thresholds
NUM_THRESHOLDS=${#THRESHOLDS[@]}
NUM_SEED_MOLS=${#SEED_MOL_INDICES[@]}
NUM_SEEDS=${#SEEDS[@]}

threshold_idx=$((TASK_ID % NUM_THRESHOLDS))
remainder=$((TASK_ID / NUM_THRESHOLDS))
seed_mol_idx=$((remainder % NUM_SEED_MOLS))
remainder=$((remainder / NUM_SEED_MOLS))
seed_idx=$((remainder % NUM_SEEDS))
target_idx=$((remainder / NUM_SEEDS))

TARGET_NAME="${TARGETS[$target_idx]}"
SEED="${SEEDS[$seed_idx]}"
SEED_MOL_IDX="${SEED_MOL_INDICES[$seed_mol_idx]}"
THRESHOLD="${THRESHOLDS[$threshold_idx]}"

# Seed molecules for each target (extracted from actives.csv - 3 molecules per target)
# Format: ['target_0'], ['target_1'], ['target_2'] for the 3 seed molecules per target
declare -A SEED_MOLECULES=(
    # parp1 (3 seed molecules)
    ['parp1_0']='CN(C)Cc3ccc2c(CNC(=O)c1cccn12)c3'
    ['parp1_1']='COc1[nH]c3cccc2C(=O)NCCc1c23'
    ['parp1_2']='O/N=C/c1cn3CCNC(=O)c2cccc1c23'
    # fa7 (3 seed molecules)
    ['fa7_0']='CC(C)CCN(Cc2ccc1ccc(C(N)=N)cc1c2)C(=O)c3cccc4ccccc34'
    ['fa7_1']='N[C@H](Cc1ccccc1)C(=O)N2CCC[C@H]2C(=O)N[C@H](CCl)CCCN=C(N)N'
    ['fa7_2']='CC(C)Nc3ccc(c1cc(N)cc(C(O)=O)c1)n(CC(=O)NCc2ccc(C(N)=N)cc2)c3=O'
    # 5ht1b (3 seed molecules)
    ['5ht1b_0']='Cc1nc(-c2ccc(-c3ccc(C(=O)N4CCc5cc6c(cc54)[C@]4(CC[N@H+](C)CC4)CO6)cc3)c(C)c2)no1'
    ['5ht1b_1']='FC(F)(F)c1cccc(N2CC[NH2+]CC2)c1'
    ['5ht1b_2']='C1=CC2=NC=C(CCCN3CC[NH+](CCc4ccccc4)CC3)[C@H]2C=C1n1cnnc1'
)

SEED_MOLECULE="${SEED_MOLECULES[${TARGET_NAME}_${SEED_MOL_IDX}]}"

if [ -z "$SEED_MOLECULE" ]; then
    echo "Error: Seed molecule not found for target $TARGET_NAME index $SEED_MOL_IDX"
    exit 1
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Task ID: $TASK_ID"
echo "Target: $TARGET_NAME (index $target_idx)"
echo "Seed: $SEED (index $seed_idx)"
echo "Seed Molecule Index: $SEED_MOL_IDX"
echo "Threshold: $THRESHOLD (index $threshold_idx)"
echo "Seed Molecule: $SEED_MOLECULE"
echo "Method: genetic_gfn"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gflownet-pmo

# Change to lead_task directory (in repo)
cd "${MULTI_OBJ_DIR}/lead_task"

# ---- MPS for better GPU sharing (A100) ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

# Avoid CPU oversubscription
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

# Configuration
OBJECTIVES="qed,sa,${TARGET_NAME}"
ALPHA_VECTOR="1,1,1"
MAX_ORACLE_CALLS=3000
BASE_OUT="${RESULTS_ROOT}/${TARGET_NAME}_lead_task_seed${SEED}_mol${SEED_MOL_IDX}_th${THRESHOLD}"
mkdir -p "$BASE_OUT"

# Use simple task mode with single seed (not production mode)
# Note: Threshold is included in output directory name for organization
# The actual threshold filtering is done in post-processing (eval.py)
srun --exclusive -n1 bash -lc "
  python run.py genetic_gfn \
    --objectives ${OBJECTIVES} \
    --alpha_vector ${ALPHA_VECTOR} \
    --seed_molecule \"${SEED_MOLECULE}\" \
    --max_oracle_calls ${MAX_ORACLE_CALLS} \
    --seed ${SEED} \
    --task simple \
    --output_dir ${BASE_OUT} \
    --run_name ${TARGET_NAME}_lead_task_seed${SEED}_mol${SEED_MOL_IDX}_th${THRESHOLD} \
    --wandb disabled \
    --freq_log 100
"

# ---- Stop MPS ----
echo quit | nvidia-cuda-mps-control
rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "=========================================="
echo "End time: $(date)"
echo "Target $TARGET_NAME (seed $SEED, mol $SEED_MOL_IDX, threshold $THRESHOLD) completed"
echo "CSV files saved in: ${BASE_OUT}/genetic_gfn/results/"
echo "=========================================="
