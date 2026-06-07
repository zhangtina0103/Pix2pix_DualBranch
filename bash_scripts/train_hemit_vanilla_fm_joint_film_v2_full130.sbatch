#!/bin/bash
# FiLM v2: 15-epoch finetune from joint_perc_full/130 (~2–3h @ bs=4 512²).
#
#   sbatch bash_scripts/train_hemit_vanilla_fm_joint_film_v2_full130.sbatch
#   export TEST_EPOCH=145 && sbatch bash_scripts/test_hemit_vanilla_fm_joint_film_v2.sbatch
#     (set TRAIN_NAME=hemit_vanilla_fm_joint_film_v2_ft130 in test or use eval wrapper)
#SBATCH --job-name=hemit_film2ft
#SBATCH --output=logs/hemit_vfm_joint_film_v2_full130_%j.out
#SBATCH --error=logs/hemit_vfm_joint_film_v2_full130_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs
source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL=vanilla_fm MODE=train
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export HEMIT_TRAIN_SIZE=512
export BATCH_SIZE=4
export DISPLAY_ID=-1

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_joint_film_v2_env
export TRAIN_NAME=hemit_vanilla_fm_joint_film_v2_ft130
export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
export VANILLA_FM_ENV_LOCKED=1
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_finetune_from_joint_perc.sh"
vanilla_fm_finetune_from_joint_perc_full130 15
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
