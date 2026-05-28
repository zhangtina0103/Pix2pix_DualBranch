#!/bin/bash
# One-time: create hemit_compare env (CUT/ASP/CycleGAN/FM). Does not modify pix2pix_cuda.
#SBATCH --job-name=setup_hemit_cmp
#SBATCH --output=logs/setup_hemit_compare_%j.out
#SBATCH --error=logs/setup_hemit_compare_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs
bash scripts/setup_hemit_compare_env.sh
