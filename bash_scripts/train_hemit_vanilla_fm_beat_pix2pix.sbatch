#!/bin/bash
# Vanilla FM (Gaussian noise + Heun ODE) tuned for SSIM — still NOT pix2pix G(A).
# Mentor core + optional ODE sample L1. PatchGAN off unless FM_USE_GAN=1.
#
#   sbatch bash_scripts/train_hemit_vanilla_fm_beat_pix2pix.sbatch
# Strict mentor only: source vanilla_fm_env.sh && vanilla_fm_apply_mentor_strict_env && ...
#
# OOM → FM_PERC_SIZE=256 FM_SAMPLE_L1_PROB=0.25 BATCH_SIZE=1
#SBATCH --job-name=hemit_vfm_p2p
#SBATCH --output=logs/hemit_vfm_beat_p2p_%j.out
#SBATCH --error=logs/hemit_vfm_beat_p2p_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export DISPLAY_ID="${DISPLAY_ID:--1}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_beat_pix2pix_env
export VANILLA_FM_ENV_LOCKED=1
vanilla_fm_print_train_env

bash scripts/run_hemit_vanilla_fm.sh
