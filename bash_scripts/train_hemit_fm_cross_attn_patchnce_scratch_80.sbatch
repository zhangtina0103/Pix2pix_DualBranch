#!/bin/bash
# Cross-attn + PatchNCE, scratch 1→80 @ 512² bs=4 (6h walltime chunks).
#
#   sbatch bash_scripts/train_hemit_fm_cross_attn_patchnce_scratch_80.sbatch
# Resume until epoch 80 (~25–30 ep/chunk @ ~12–14 min/ep):
#   export RESUME_PROFILE=cross_attn_patchnce
#   sbatch bash_scripts/resume_hemit_fm_scratch.sbatch
# Eval:
#   export TEST_EPOCH=80
#   sbatch bash_scripts/eval_hemit_fm_cross_attn_patchnce_scratch.sbatch
#SBATCH --job-name=hemit_xa_nce
#SBATCH --output=logs/hemit_fm_cross_attn_patchnce_scratch_%j.out
#SBATCH --error=logs/hemit_fm_cross_attn_patchnce_scratch_%j.err
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

export MODEL=vanilla_fm MODE=train
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export HEMIT_TRAIN_SIZE=512
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID=-1

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_fm_cross_attn_patchnce_scratch_env
unset FM_CROSS_ATTN_DECODER
export FM_CROSS_ATTN_HEADS="${FM_CROSS_ATTN_HEADS:-4}"
export VANILLA_FM_ENV_LOCKED=1
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
