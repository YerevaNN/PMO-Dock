#!/bin/bash
# REINVENT4 on hit (or spec) tasks.
#   sbatch reinvent4/slurm/hit.slurm <oracle_url> [extra runner args...]
# e.g.
#   sbatch reinvent4/slurm/hit.slurm http://172.26.26.195:5000
#   sbatch reinvent4/slurm/hit.slurm http://172.26.26.195:5000 --hparam_config hit/hparams.yaml
#
# Start the docking service first; REINVENT4 has no inline docking (no OpenBabel in its env).
#SBATCH --job-name=reinvent4_hit
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/reinvent4_hit/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/reinvent4_hit/%j.err

set -euo pipefail

if [[ -z "${1:-}" ]]; then
  echo "Usage: sbatch reinvent4/slurm/hit.slurm <oracle_url> [extra runner args...]"
  exit 2
fi
ORACLE_URL="$1"; shift

export PROJECT_ROOT="${PROJECT_ROOT:-/home/gsimonyan/PMO-Dock}"
export WEKA_ROOT="${WEKA_ROOT:-/mnt/weka/$USER}"
export OUT_DIR="${OUT_DIR:-$WEKA_ROOT/results}"
export REINVENT_PRIOR_BASE="${REINVENT_PRIOR_BASE:-$WEKA_ROOT/large-assets/reinvent4/src/priors}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/reinvent4/plugins:${PYTHONPATH:-}"
export DOCKING_VINA_URL="$ORACLE_URL"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate "$WEKA_ROOT/envs/reinvent4"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "python: $(command -v python)"
echo "oracle: $DOCKING_VINA_URL"
echo "out:    $OUT_DIR"

cd "$PROJECT_ROOT"
python reinvent4/reinvent_hit_runner.py \
  --config_file reinvent4/configs/hit.json \
  --oracle_name parp1 fa7 5ht1b braf jak2 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --max_oracle_calls 3000 \
  --reward_type geam \
  --oracle_url "$ORACLE_URL" \
  --max_workers 4 \
  "$@"
