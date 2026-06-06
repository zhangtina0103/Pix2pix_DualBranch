#!/bin/bash
# MONAI UNet @ 512² + mid attention — noise path only, scratch 1→130 (6h chunks).
#
#   sbatch bash_scripts/train_hemit_vanilla_fm_monai512.sbatch
# Resume (6h chunks; keep BATCH_SIZE=2 for MONAI attn):
#   unset RESUME_FROM_EPOCH EPOCH_COUNT
#   export RESUME_PROFILE=monai512 BATCH_SIZE=2
#   sbatch bash_scripts/resume_hemit_fm_scratch.sbatch
# Eval @ 945:
#   export TEST_EPOCH=130
#   sbatch bash_scripts/eval_hemit_vanilla_fm_monai512.sbatch
#SBATCH --job-name=hemit_monai512
#SBATCH --output=logs/hemit_monai512_%j.out
#SBATCH --error=logs/hemit_monai512_%j.err
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
export BATCH_SIZE="${BATCH_SIZE:-2}"
export DISPLAY_ID=-1
export NUM_THREADS=8

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_monai512_env
export VANILLA_FM_ENV_LOCKED=1
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
