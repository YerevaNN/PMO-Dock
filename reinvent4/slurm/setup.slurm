#!/bin/bash
# One-off: build the reinvent4 env and verify the checkout + priors.
#   sbatch reinvent4/slurm/setup.slurm
#
# Must go through SLURM: the conda solver needs several GB and login-node shells here
# are cgroup-capped at 4 GiB, where it is OOM-killed collecting package metadata.
# No GPU needed -- this only installs.
#SBATCH --job-name=reinvent4_setup
#SBATCH --partition=research_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/weka/gsimonyan/results/slurm_logs/reinvent4_setup/%j.out
#SBATCH --error=/mnt/weka/gsimonyan/results/slurm_logs/reinvent4_setup/%j.err

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/gsimonyan/PMO-Dock}"
cd "$PROJECT_ROOT"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}"
bash reinvent4/env/setup.sh
