#!/bin/bash
# Ablation 1: bilinear decoder only — same FM as joint_perc (x1 + perc 0.1, 25 Heun, no ODE-L1/GAN).
#
#   sbatch bash_scripts/train_hemit_vanilla_fm_decoder_only.sbatch
#SBATCH --job-name=hemit_vfm_dec
#SBATCH --output=logs/hemit_vfm_decoder_%j.out
#SBATCH --error=logs/hemit_vfm_decoder_%j.err
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
vanilla_fm_apply_decoder_only_env
export VANILLA_FM_ENV_LOCKED=1
export TRAIN_NAME=hemit_vanilla_fm_decoder_only
vanilla_fm_print_train_env

bash scripts/run_hemit_vanilla_fm.sh
