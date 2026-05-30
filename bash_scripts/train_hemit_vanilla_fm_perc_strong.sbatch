#!/bin/bash
# Ablation 2: stronger perceptual (default 0.5) — vanilla FM, parallel to decoder_only.
#
#   sbatch bash_scripts/train_hemit_vanilla_fm_perc_strong.sbatch
# Tune: FM_LAMBDA_PERC=1.0 FM_PERC_SIZE=256  (OOM → keep 0.5 or PERC_SIZE=128)
#SBATCH --job-name=hemit_vfm_ps
#SBATCH --output=logs/hemit_vfm_perc_%j.out
#SBATCH --error=logs/hemit_vfm_perc_%j.err
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
vanilla_fm_apply_perc_strong_env
export VANILLA_FM_ENV_LOCKED=1
export TRAIN_NAME=hemit_vanilla_fm_perc_strong
export FM_CHANNEL_WEIGHTS=1,1,1
vanilla_fm_print_train_env

bash scripts/run_hemit_vanilla_fm.sh
