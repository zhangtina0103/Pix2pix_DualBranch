#!/bin/bash
# One CaMSC fold per array task (fits 6h wall limit; run 5 folds in parallel).
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   sbatch bash_scripts/train_camsc_bf_fm_cross_attn_kfold_array.sbatch
#SBATCH --job-name=camsc_xa_kf
#SBATCH --output=logs/camsc_fm_cross_attn_%A_%a.out
#SBATCH --error=logs/camsc_fm_cross_attn_%A_%a.err
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
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"

export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
camsc_apply_train_data_env

export CAMSC_MODEL=fm_cross_attn
export MODE=train
export GPU_IDS=0
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID="${DISPLAY_ID:--1}"
export N_EPOCHS="${N_EPOCHS:-100}"
export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-20}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-100}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"

camsc_apply_model_env

FOLD="${SLURM_ARRAY_TASK_ID}"
echo "fm_cross_attn fold ${FOLD} @ ${CAMSC_KFOLD_ROOT}"
vanilla_fm_print_train_env
camsc_run_fold_train "${FOLD}"
