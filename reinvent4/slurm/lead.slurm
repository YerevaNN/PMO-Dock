#!/bin/bash
# REINVENT4 on lead tasks with a seeded generator -- the run that distinguishes
# REINVENT4 from everything else in the benchmark.
#   sbatch reinvent4/slurm/lead.slurm <oracle_url> [extra runner args...]
# e.g.
#   sbatch reinvent4/slurm/lead.slurm http://172.26.26.195:5000
#   sbatch reinvent4/slurm/lead.slurm http://172.26.26.195:5000 --sim_thresholds 0.4
#   sbatch reinvent4/slurm/lead.slurm http://172.26.26.195:5000 --generator libinvent \
#          --config_file reinvent4/configs/lead_libinvent.json
#
# Defaults to Mol2Mol at similarity tier 06 (prior .m2m_high; tier 04 -> .m2m_medium).
#SBATCH --job-name=reinvent4_lead
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/reinvent4_lead/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/reinvent4_lead/%j.err

set -euo pipefail

if [[ -z "${1:-}" ]]; then
  echo "Usage: sbatch reinvent4/slurm/lead.slurm <oracle_url> [extra runner args...]"
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
python reinvent4/reinvent_lead_runner.py \
  --config_file reinvent4/configs/lead_m2m.json \
  --generator mol2mol \
  --oracle_name parp1 fa7 5ht1b braf jak2 \
  --start_mol_idx 0 1 2 \
  --seeds 0 1 2 \
  --sim_thresholds 0.6 \
  --max_oracle_calls 1000 \
  --reward_type geam \
  --oracle_url "$ORACLE_URL" \
  --max_workers 4 \
  "$@"
