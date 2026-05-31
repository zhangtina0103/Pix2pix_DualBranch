#!/bin/bash
# Mentor: joint_perc + PatchNCE (paired H&E ↔ x1_hat). Finetune 81→100.
#   sbatch bash_scripts/train_hemit_vanilla_fm_joint_perc_patchnce.sbatch
#SBATCH --job-name=hemit_perc_nce
#SBATCH --output=logs/hemit_perc_patchnce_%j.out
#SBATCH --error=logs/hemit_perc_patchnce_%j.err
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

export MODEL=vanilla_fm
export MODE=train
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export BATCH_SIZE=1
export DISPLAY_ID=-1
export TRAIN_LR=0.0001

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_joint_perc_patchnce_env
export VANILLA_FM_ENV_LOCKED=1
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_finetune_from_joint_perc.sh"
vanilla_fm_finetune_from_joint_perc_80
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
