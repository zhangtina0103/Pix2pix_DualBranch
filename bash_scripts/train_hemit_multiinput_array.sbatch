#!/bin/bash
# Train all three HEMIT multi-input variants (array 0=he, 1=dapi, 2=he_dapi).
#
#   sbatch bash_scripts/prepare_hemit_multiinput.sbatch
#   sbatch bash_scripts/train_hemit_multiinput_array.sbatch
#SBATCH --job-name=hemit_multi_arr
#SBATCH --output=logs/hemit_multiinput_train_%A_%a.out
#SBATCH --error=logs/hemit_multiinput_train_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-2
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

case "${SLURM_ARRAY_TASK_ID}" in
  0) export HEMIT_MULTI_VARIANT=he ;;
  1) export HEMIT_MULTI_VARIANT=dapi ;;
  2) export HEMIT_MULTI_VARIANT=he_dapi ;;
  *) echo "ERROR: bad array id ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

export MODEL=vanilla_fm
export MODE=train
export HEMIT_TRAIN_SIZE="${HEMIT_TRAIN_SIZE:-512}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GPU_IDS="${GPU_IDS:-0}"
export DISPLAY_ID="${DISPLAY_ID:--1}"

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hemit_multiinput_env.sh"
hemit_multiinput_apply_env

[[ -d "${DATAROOT}/trainA" ]] || {
  echo "ERROR: missing ${DATAROOT}/trainA — run prepare_hemit_multiinput.sbatch first" >&2
  echo "  HEMIT_MULTI_VARIANTS=${HEMIT_MULTI_VARIANT} sbatch bash_scripts/prepare_hemit_multiinput.sbatch" >&2
  exit 1
}

vanilla_fm_print_train_env
echo "array=${SLURM_ARRAY_TASK_ID} HEMIT_MULTI_VARIANT=${HEMIT_MULTI_VARIANT}"
echo "DATAROOT=${DATAROOT}"
echo "CHECKPOINTS_DIR=${CHECKPOINTS_DIR}"
ls "${DATAROOT}/trainA" | wc -l

bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
