#!/bin/bash
# One-time: GPU PyTorch for pix2pix env. Wait for "OK" in log, then run_hemit_reproduce.sbatch
#
#SBATCH --job-name=setup_pix2pix
#SBATCH --output=logs/setup_pix2pix_%j.out
#SBATCH --error=logs/setup_pix2pix_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs
bash scripts/setup_pix2pix_env.sh
echo "Next: sbatch bash_scripts/run_hemit_reproduce.sbatch"
