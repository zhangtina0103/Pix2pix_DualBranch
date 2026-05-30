#!/bin/bash
# Finetune joint_perc @ 80 with FiLM on decoder (H&E -> gamma/beta per up level).
#
#   sbatch bash_scripts/train_hemit_vanilla_fm_joint_film.sbatch
#SBATCH --job-name=hemit_vfm_film
#SBATCH --output=logs/hemit_vfm_joint_film_%j.out
#SBATCH --error=logs/hemit_vfm_joint_film_%j.err
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

export CONTINUE_TRAIN=1
export RESUME_FROM_EPOCH=80
export EPOCH_COUNT=81
export N_EPOCHS=70
export N_EPOCHS_DECAY=30

# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_joint_film_env
export VANILLA_FM_ENV_LOCKED=1

mkdir -p "${REPO_ROOT}/checkpoints/${TRAIN_NAME}"
cp -f "${REPO_ROOT}/checkpoints/hemit_vanilla_fm_joint_perc/80_net_G.pth" \
  "${REPO_ROOT}/checkpoints/${TRAIN_NAME}/80_net_G.pth"

vanilla_fm_print_train_env
bash scripts/run_hemit_vanilla_fm.sh
