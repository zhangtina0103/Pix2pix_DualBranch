#!/bin/bash
# Pilot: finetune HEMIT GAN baseline → CaMSC fold 0 (fair match vs fm_cross_attn_ft).
# Default: pix2pix (strongest CaMSC scratch baseline). Use cut_ft or asp_ft to compare others.
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   sbatch bash_scripts/train_camsc_bf_gan_finetune_fold0.sbatch
#   CAMSC_GAN_FT_MODEL=cut_ft sbatch bash_scripts/train_camsc_bf_gan_finetune_fold0.sbatch
#
# After train:
#   CAMSC_GAN_FT_MODEL=pix2pix_ft CAMSC_EVAL_FOLDS=0 TEST_EPOCH=110 sbatch bash_scripts/eval_camsc_bf_gan_finetune.sbatch
#SBATCH --job-name=camsc_gan_ft0
#SBATCH --output=logs/camsc_gan_ft_fold0_%j.out
#SBATCH --error=logs/camsc_gan_ft_fold0_%j.err
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

# camsc_scratch_env.sh defaults CAMSC_MODEL=vanilla_fm — override for this job.
export CAMSC_MODEL="${CAMSC_GAN_FT_MODEL:-pix2pix_ft}"
case "${CAMSC_MODEL}" in
  pix2pix_ft|cut_ft|asp_ft) ;;
  *) echo "ERROR: CAMSC_MODEL=${CAMSC_MODEL} (use pix2pix_ft|cut_ft|asp_ft or set CAMSC_GAN_FT_MODEL)" >&2; exit 1 ;;
esac

export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
camsc_apply_train_data_env

export MODE=train
export GPU_IDS=0
export DISPLAY_ID="${DISPLAY_ID:--1}"
export HEMIT_GAN_SRC_EPOCH="${HEMIT_GAN_SRC_EPOCH:-80}"
export CAMSC_FT_EXTRA_EPOCHS="${CAMSC_FT_EXTRA_EPOCHS:-30}"
export CAMSC_FT_DECAY_EPOCHS="${CAMSC_FT_DECAY_EPOCHS:-0}"
export TRAIN_LR="${TRAIN_LR:-5e-5}"

if [[ "${CAMSC_MODEL}" == cut_ft || "${CAMSC_MODEL}" == asp_ft ]]; then
  export BATCH_SIZE="${BATCH_SIZE:-2}"
else
  export BATCH_SIZE="${BATCH_SIZE:-4}"
fi

camsc_apply_model_env

FOLD=0
echo "${CAMSC_MODEL} fold ${FOLD} @ ${CAMSC_KFOLD_ROOT}"
echo "  init epoch ${HEMIT_GAN_SRC_EPOCH}, +${CAMSC_FT_EXTRA_EPOCHS} finetune epochs, lr=${TRAIN_LR}"
camsc_run_fold_train "${FOLD}"
