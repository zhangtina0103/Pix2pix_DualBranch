#!/bin/bash
# One-time: pip venv at .venv-hemit (README-style, no conda).
#
#SBATCH --job-name=setup_hemit_venv
#SBATCH --output=logs/setup_hemit_venv_%j.out
#SBATCH --error=logs/setup_hemit_venv_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs
bash scripts/setup_venv_engaging.sh
echo "Next: sbatch bash_scripts/run_hemit_reproduce.sbatch"
