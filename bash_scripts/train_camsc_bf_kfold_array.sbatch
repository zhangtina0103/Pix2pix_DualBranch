#!/bin/bash
# Train vanilla FM on one CaMSC k-fold split per array task (fold 0..4).
#
#   sbatch bash_scripts/prepare_camsc_bf.sbatch
#   sbatch bash_scripts/train_camsc_bf_kfold_array.sbatch
#
# Test + aggregate after training:
#   sbatch bash_scripts/eval_camsc_kfold_array.sbatch
#SBATCH --job-name=camsc_kf
#SBATCH --output=logs/camsc_kfold_train_%A_%a.out
#SBATCH --error=logs/camsc_kfold_train_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-4
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

FOLD="${SLURM_ARRAY_TASK_ID}"
export DATAROOT
DATAROOT="$(camsc_fold_dataroot "${FOLD}")"
export TRAIN_NAME
TRAIN_NAME="$(camsc_fold_train_name "${FOLD}")"
export MODEL="${CAMSC_MODEL:-vanilla_fm}"
export MODE=train
export GPU_IDS=0
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID="${DISPLAY_ID:--1}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"
export N_EPOCHS="${N_EPOCHS:-80}"
export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-20}"

[[ -d "${DATAROOT}/trainA" ]] || {
  echo "ERROR: missing ${DATAROOT}/trainA — run prepare_camsc_bf.sbatch first" >&2
  exit 1
}

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_train_env
vanilla_fm_print_train_env

echo "fold=${FOLD} DATAROOT=${DATAROOT} TRAIN_NAME=${TRAIN_NAME}"
ls "${DATAROOT}/trainA" | wc -l
bash scripts/run_hemit_vanilla_fm.sh
