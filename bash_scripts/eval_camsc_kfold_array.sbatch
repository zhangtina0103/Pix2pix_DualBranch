#!/bin/bash
# Test vanilla FM for each CaMSC k-fold + aggregate Hoechst/WT1 metrics.
#
#   TEST_EPOCH=80 sbatch bash_scripts/eval_camsc_kfold_array.sbatch
#SBATCH --job-name=camsc_ev
#SBATCH --output=logs/camsc_kfold_eval_%A_%a.out
#SBATCH --error=logs/camsc_kfold_eval_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

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
export MODE=test
export GPU_IDS=0
export TEST_EPOCH="${TEST_EPOCH:-80}"
export NUM_TEST="${NUM_TEST:-9999}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_train_env

echo "fold=${FOLD} test epoch=${TEST_EPOCH} name=${TRAIN_NAME}"
bash scripts/run_hemit_vanilla_fm.sh

# Aggregate only on last array task (fold 4)
if [[ "${FOLD}" == "$((CAMSC_KFOLDS - 1))" ]]; then
  python scripts/eval_camsc_kfold.py \
    --kfold-root "${CAMSC_KFOLD_ROOT}" \
    --results-root "${REPO_ROOT}/results" \
    --name-prefix "camsc_bf_${CAMSC_MODEL:-vanilla_fm}_fold" \
    --name-suffix "_${HEMIT_TRAIN_SIZE:-512}" \
    --epoch "${TEST_EPOCH}" \
    --k-folds "${CAMSC_KFOLDS}"
fi
