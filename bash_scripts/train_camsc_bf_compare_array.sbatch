#!/bin/bash
# CaMSC k-fold: pix2pix + FM cross-attn in parallel (2 GPUs).
#
#   sbatch bash_scripts/prepare_camsc_bf.sbatch
#   sbatch bash_scripts/train_camsc_bf_compare_array.sbatch
#   TEST_EPOCH=50 sbatch bash_scripts/eval_camsc_bf_compare.sbatch
#
# Array task 0 = pix2pix (ResNet9, ~11M G)
# Array task 1 = fm_cross_attn (flow matching + H&E cross-attn bottleneck)
#SBATCH --job-name=camsc_cmp
#SBATCH --output=logs/camsc_compare_%A_%a.out
#SBATCH --error=logs/camsc_compare_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-1
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

export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
camsc_apply_train_data_env

case "${SLURM_ARRAY_TASK_ID}" in
  0) export CAMSC_MODEL=pix2pix ;;
  1) export CAMSC_MODEL=fm_cross_attn ;;
  *) echo "ERROR: unknown array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

echo "SLURM array task ${SLURM_ARRAY_TASK_ID} → CAMSC_MODEL=${CAMSC_MODEL}"
echo "K-fold root: ${CAMSC_KFOLD_ROOT}"

export MODE=train
export GPU_IDS=0
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID="${DISPLAY_ID:--1}"
export N_EPOCHS="${N_EPOCHS:-100}"
export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-20}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-100}"

if [[ "${CAMSC_MODEL}" == "fm_cross_attn" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
fi

camsc_apply_model_env

if [[ "${CAMSC_MODEL}" == "pix2pix" ]]; then
  echo "pix2pix: ResNet9 G, names like camsc_bf_pix2pix_fold{N}_512"
else
  vanilla_fm_print_train_env
  echo "fm_cross_attn: flow matching + cross-attn bottleneck"
fi

for FOLD in $(seq 0 $((CAMSC_KFOLDS - 1))); do
  [[ -d "$(camsc_fold_dataroot "${FOLD}")/trainA" ]] || {
    echo "ERROR: missing fold data — run prepare_camsc_bf.sbatch" >&2
    exit 1
  }
  echo ""
  echo "========== ${CAMSC_MODEL} fold ${FOLD}/${CAMSC_KFOLDS} =========="
  camsc_run_fold_train "${FOLD}"
done

echo "Done: ${CAMSC_MODEL} all ${CAMSC_KFOLDS} folds."
