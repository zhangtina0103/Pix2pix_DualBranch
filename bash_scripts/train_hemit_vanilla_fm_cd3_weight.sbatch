#!/bin/bash
# Ablation 3: CD3 channel weight 2x in x1 L1 (1,2,1). Vanilla FM, parallel if you have a 3rd GPU.
#
#   sbatch bash_scripts/train_hemit_vanilla_fm_cd3_weight.sbatch
#SBATCH --job-name=hemit_vfm_c3
#SBATCH --output=logs/hemit_vfm_cd3_%j.out
#SBATCH --error=logs/hemit_vfm_cd3_%j.err
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
vanilla_fm_apply_cd3_weight_env
export VANILLA_FM_ENV_LOCKED=1
export TRAIN_NAME=hemit_vanilla_fm_cd3_weight
vanilla_fm_print_train_env

bash scripts/run_hemit_vanilla_fm.sh
