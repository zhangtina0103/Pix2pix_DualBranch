#!/bin/bash
# Train one HEMIT multi-input variant on scratch (FM + cross-attn, 80 epochs).
#
#   sbatch bash_scripts/prepare_hemit_multiinput.sbatch
#   HEMIT_MULTI_VARIANT=he sbatch bash_scripts/train_hemit_multiinput.sbatch
#   HEMIT_MULTI_VARIANT=dapi sbatch bash_scripts/train_hemit_multiinput.sbatch
#   HEMIT_MULTI_VARIANT=he_dapi sbatch bash_scripts/train_hemit_multiinput.sbatch
#
# All three: sbatch bash_scripts/train_hemit_multiinput_array.sbatch
#SBATCH --job-name=hemit_multi_tr
#SBATCH --output=logs/hemit_multiinput_train_%j.out
#SBATCH --error=logs/hemit_multiinput_train_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL=vanilla_fm
export MODE=train
export HEMIT_MULTI_VARIANT="${HEMIT_MULTI_VARIANT:-he}"
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
echo "HEMIT_MULTI_VARIANT=${HEMIT_MULTI_VARIANT}"
echo "DATAROOT=${DATAROOT}"
echo "CHECKPOINTS_DIR=${CHECKPOINTS_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"
ls "${DATAROOT}/trainA" | wc -l

bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
