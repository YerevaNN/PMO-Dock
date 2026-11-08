#!/bin/bash
# Run the REINVENT4 integration test suite on a compute node.
#   sbatch reinvent4/slurm/tests.slurm
#
# Worth running here rather than on the login node: with REINVENT4 installed the suite
# also validates the shipped configs against upstream's pydantic schema and checks that
# every prior the configs name resolves on disk.
#SBATCH --job-name=reinvent4_tests
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/reinvent4_tests/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/reinvent4_tests/%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/gsimonyan/PMO-Dock}"
export WEKA_ROOT="${WEKA_ROOT:-/mnt/weka/$USER}"
export REINVENT_PRIOR_BASE="${REINVENT_PRIOR_BASE:-$WEKA_ROOT/large-assets/reinvent4/src/priors}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/reinvent4/plugins:${PYTHONPATH:-}"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate "$WEKA_ROOT/envs/reinvent4"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
echo "python: $(command -v python)"

cd "$PROJECT_ROOT"
python -m pytest reinvent4/tests -q "$@"
