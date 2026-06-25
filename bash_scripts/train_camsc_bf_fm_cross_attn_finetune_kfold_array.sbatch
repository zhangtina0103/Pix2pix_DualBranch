#!/bin/bash
# Finetune HEMIT cross-attn → CaMSC (one fold per array task). NOT scratch.
# Default array: folds 1-4. Full: sbatch --array=0-4 ...
#
#   sbatch --array=1-4 bash_scripts/train_camsc_bf_fm_cross_attn_finetune_kfold_array.sbatch
#SBATCH --job-name=camsc_xa_ft
#SBATCH --output=logs/camsc_fm_cross_attn_ft_%A_%a.out
#SBATCH --error=logs/camsc_fm_cross_attn_ft_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=1-4
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

export HEMIT_CROSS_ATTN_SRC_EPOCH="${HEMIT_CROSS_ATTN_SRC_EPOCH:-80}"
export CAMSC_FT_EXTRA_EPOCHS="${CAMSC_FT_EXTRA_EPOCHS:-30}"
export CAMSC_FT_DECAY_EPOCHS="${CAMSC_FT_DECAY_EPOCHS:-0}"
export TRAIN_LR="${TRAIN_LR:-5e-5}"

camsc_apply_model_env

FOLD="${SLURM_ARRAY_TASK_ID}"
FT_END=$((HEMIT_CROSS_ATTN_SRC_EPOCH + CAMSC_FT_EXTRA_EPOCHS + CAMSC_FT_DECAY_EPOCHS))
RESOLVED="$(camsc_resolved_train_name "${FOLD}")"
if [[ "${SKIP_IF_DONE:-1}" == "1" ]] && [[ -f "${REPO_ROOT}/checkpoints/${RESOLVED}/${FT_END}_net_G.pth" ]]; then
  echo "[skip] fm_cross_attn_ft fold ${FOLD}: already at epoch ${FT_END}"
  exit 0
fi
echo "FINETUNE fm_cross_attn_ft fold ${FOLD} (HEMIT@${HEMIT_CROSS_ATTN_SRC_EPOCH} → ${FT_END}) ckpt=${RESOLVED}"
vanilla_fm_print_train_env
camsc_run_fold_train "${FOLD}"
