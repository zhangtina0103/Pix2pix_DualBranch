#!/bin/bash
# Pilot: finetune HEMIT cross-attn → CaMSC fold 0 only (~6h).
# Gate: eval vs pix2pix fold0 (0.758 Hoechst / 0.834 WT1) before running all folds.
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   sbatch bash_scripts/train_camsc_bf_fm_cross_attn_finetune_fold0.sbatch
#
# After train:
#   TEST_EPOCH=110 sbatch bash_scripts/eval_camsc_bf_fm_cross_attn_finetune.sbatch
#   (or TEST_EPOCH=95,100,105 if earlier checkpoint is better)
#SBATCH --job-name=camsc_xa_ft0
#SBATCH --output=logs/camsc_fm_cross_attn_ft_fold0_%j.out
#SBATCH --error=logs/camsc_fm_cross_attn_ft_fold0_%j.err
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

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/camsc_scratch_env.sh"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"

export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
camsc_apply_train_data_env

export CAMSC_MODEL=fm_cross_attn_ft
export MODE=train
export GPU_IDS=0
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID="${DISPLAY_ID:--1}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"

# HEMIT init @80 → train epochs 81..110 (30 ep, lr=5e-5)
export HEMIT_CROSS_ATTN_SRC_EPOCH="${HEMIT_CROSS_ATTN_SRC_EPOCH:-80}"
export CAMSC_FT_EXTRA_EPOCHS="${CAMSC_FT_EXTRA_EPOCHS:-30}"
export CAMSC_FT_DECAY_EPOCHS="${CAMSC_FT_DECAY_EPOCHS:-0}"
export TRAIN_LR="${TRAIN_LR:-5e-5}"

camsc_apply_model_env
vanilla_fm_print_train_env

FOLD=0
echo "fm_cross_attn_ft fold ${FOLD} @ ${CAMSC_KFOLD_ROOT}"
echo "  init: ${HEMIT_CROSS_ATTN_SRC_NAME:-hemit_fm_cross_attn_scratch_512}@${HEMIT_CROSS_ATTN_SRC_EPOCH}"
echo "  finetune: +${CAMSC_FT_EXTRA_EPOCHS} epochs from $((${HEMIT_CROSS_ATTN_SRC_EPOCH} + 1))"
camsc_run_fold_train "${FOLD}"
