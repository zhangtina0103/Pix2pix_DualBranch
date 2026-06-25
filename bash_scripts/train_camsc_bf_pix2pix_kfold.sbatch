#!/bin/bash
# CaMSC k-fold pix2pix with augmentations (full-res prep + random 512 crops).
#
#   CAMSC_ENABLE_AUG=1 sbatch bash_scripts/prepare_camsc_bf.sbatch
#   sbatch bash_scripts/train_camsc_bf_pix2pix_kfold.sbatch
#SBATCH --job-name=camsc_p2p
#SBATCH --output=logs/camsc_pix2pix_%j.out
#SBATCH --error=logs/camsc_pix2pix_%j.err
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

export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
camsc_apply_train_data_env

export CAMSC_MODEL=pix2pix
export MODE=train
export GPU_IDS=0
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID="${DISPLAY_ID:--1}"
export N_EPOCHS="${N_EPOCHS:-50}"
export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-10}"

camsc_apply_model_env
echo "pix2pix k-fold @ ${CAMSC_KFOLD_ROOT}"

for FOLD in $(seq 0 $((CAMSC_KFOLDS - 1))); do
  [[ -d "$(camsc_fold_dataroot "${FOLD}")/trainA" ]] || {
    echo "ERROR: missing $(camsc_fold_dataroot "${FOLD}")/trainA" >&2
    exit 1
  }
  camsc_run_fold_train "${FOLD}"
done

echo "Done: pix2pix all ${CAMSC_KFOLDS} folds."
