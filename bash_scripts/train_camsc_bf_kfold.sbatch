#!/bin/bash
# CaMSC k-fold: vanilla FM only (separate job from pix2pix / cross-attn compare).
#
#   sbatch bash_scripts/prepare_camsc_bf.sbatch
#   sbatch bash_scripts/train_camsc_bf_kfold.sbatch
#   TEST_EPOCH=50 sbatch bash_scripts/eval_camsc_kfold.sbatch
#
# Pix2pix + cross-attn (2 GPUs): train_camsc_bf_compare_array.sbatch
#SBATCH --job-name=camsc_vfm
#SBATCH --output=logs/camsc_kfold_train_%j.out
#SBATCH --error=logs/camsc_kfold_train_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

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

export CAMSC_MODEL=vanilla_fm
export MODE=train
export GPU_IDS=0
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID="${DISPLAY_ID:--1}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"
export N_EPOCHS="${N_EPOCHS:-100}"
export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-20}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-100}"

camsc_apply_model_env
vanilla_fm_print_train_env
echo "CAMSC_MODEL=vanilla_fm (flow matching U-Net, no cross-attn)"

for FOLD in $(seq 0 $((CAMSC_KFOLDS - 1))); do
  [[ -d "$(camsc_fold_dataroot "${FOLD}")/trainA" ]] || {
    echo "ERROR: missing fold data — run prepare_camsc_bf.sbatch first" >&2
    exit 1
  }
  echo ""
  echo "========== vanilla_fm fold ${FOLD}/${CAMSC_KFOLDS} =========="
  echo "train tiles: $(ls -1 "$(camsc_fold_dataroot "${FOLD}")/trainA" | wc -l | tr -d ' ')"
  camsc_run_fold_train "${FOLD}"
done

echo "Done: vanilla_fm all ${CAMSC_KFOLDS} folds."
