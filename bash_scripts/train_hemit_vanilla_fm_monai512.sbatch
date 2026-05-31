#!/bin/bash
# Mentor: MONAI UNet @ 512² + attention (no joint_perc checkpoint — train from scratch).
#   sbatch bash_scripts/train_hemit_vanilla_fm_monai512.sbatch
#SBATCH --job-name=hemit_monai512
#SBATCH --output=logs/hemit_monai512_%j.out
#SBATCH --error=logs/hemit_monai512_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs
source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL=vanilla_fm
export MODE=train
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export BATCH_SIZE=2
export DISPLAY_ID=-1
export TRAIN_LR=0.0002
export N_EPOCHS=80
export N_EPOCHS_DECAY=20

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_monai512_env
export VANILLA_FM_ENV_LOCKED=1
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
