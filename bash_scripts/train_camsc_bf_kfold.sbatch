#!/bin/bash
# Train all CaMSC k-folds sequentially on ONE GPU (small dataset → ~1–2 h total).
#
#   sbatch bash_scripts/prepare_camsc_bf.sbatch
#   sbatch bash_scripts/train_camsc_bf_kfold.sbatch
#   TEST_EPOCH=50 sbatch bash_scripts/eval_camsc_kfold.sbatch
#SBATCH --job-name=camsc_kf
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

export MODEL="${CAMSC_MODEL:-vanilla_fm}"
export MODE=train
export GPU_IDS=0
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID="${DISPLAY_ID:--1}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"
# Small dataset: fewer epochs is plenty (override if you want longer)
export N_EPOCHS="${N_EPOCHS:-50}"
export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-10}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_train_env
vanilla_fm_print_train_env

for FOLD in $(seq 0 $((CAMSC_KFOLDS - 1))); do
  export DATAROOT="$(camsc_fold_dataroot "${FOLD}")"
  export TRAIN_NAME="$(camsc_fold_train_name "${FOLD}")"
  [[ -d "${DATAROOT}/trainA" ]] || {
    echo "ERROR: missing ${DATAROOT}/trainA — run prepare_camsc_bf.sbatch first" >&2
    exit 1
  }
  echo ""
  echo "========== fold ${FOLD}/${CAMSC_KFOLDS}  DATAROOT=${DATAROOT}  TRAIN_NAME=${TRAIN_NAME} =========="
  echo "train tiles: $(ls -1 "${DATAROOT}/trainA" | wc -l | tr -d ' ')"
  bash scripts/run_hemit_vanilla_fm.sh
done

echo "All ${CAMSC_KFOLDS} folds done."
