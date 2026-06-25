#!/bin/bash
# CaMSC k-fold FM cross-attn with train-time augmentations (random 512 crops from full-res fields).
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   sbatch bash_scripts/train_camsc_bf_fm_cross_attn_kfold_array.sbatch   # recommended (6h/fold)
#   sbatch bash_scripts/train_camsc_bf_fm_cross_attn_kfold.sbatch         # all folds sequential
#SBATCH --job-name=camsc_xattn
#SBATCH --output=logs/camsc_fm_cross_attn_%j.out
#SBATCH --error=logs/camsc_fm_cross_attn_%j.err
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
vanilla_fm_print_train_env
echo "fm_cross_attn k-fold @ ${CAMSC_KFOLD_ROOT}"

for FOLD in $(seq 0 $((CAMSC_KFOLDS - 1))); do
  [[ -d "$(camsc_fold_dataroot "${FOLD}")/trainA" ]] || {
    echo "ERROR: missing $(camsc_fold_dataroot "${FOLD}")/trainA" >&2
    exit 1
  }
  camsc_run_fold_train "${FOLD}"
done

echo "Done: fm_cross_attn all ${CAMSC_KFOLDS} folds."
