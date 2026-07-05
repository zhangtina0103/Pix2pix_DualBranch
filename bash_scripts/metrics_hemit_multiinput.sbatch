#!/bin/bash
# Recompute extended metrics only (no inference) for one multi-input variant.
#
#   HEMIT_MULTI_VARIANT=he   TEST_EPOCH=80 sbatch bash_scripts/metrics_hemit_multiinput.sbatch
#   HEMIT_MULTI_VARIANT=dapi TEST_EPOCH=80 sbatch bash_scripts/metrics_hemit_multiinput.sbatch
#   HEMIT_MULTI_VARIANT=he_dapi TEST_EPOCH=80 sbatch bash_scripts/metrics_hemit_multiinput.sbatch
#
# All three:
#   for v in he dapi he_dapi; do
#     HEMIT_MULTI_VARIANT=$v TEST_EPOCH=80 sbatch bash_scripts/metrics_hemit_multiinput.sbatch
#   done
#SBATCH --job-name=hemit_multi_mx
#SBATCH --output=logs/hemit_multiinput_metrics_%j.out
#SBATCH --error=logs/hemit_multiinput_metrics_%j.err
#SBATCH --partition=mit_normal_gpu
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

export MODEL=vanilla_fm
export MODE=metrics
export HEMIT_MULTI_VARIANT="${HEMIT_MULTI_VARIANT:-he}"
export TEST_EPOCH="${TEST_EPOCH:-80}"
export GPU_IDS="${GPU_IDS:-0}"
export DISPLAY_ID="${DISPLAY_ID:--1}"

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hemit_multiinput_env.sh"
hemit_multiinput_apply_env
export VANILLA_FM_ENV_LOCKED=1

TEST_DIR="${RESULTS_DIR}/${TRAIN_NAME}/test_${TEST_EPOCH}"
echo "HEMIT_MULTI_VARIANT=${HEMIT_MULTI_VARIANT} TRAIN_NAME=${TRAIN_NAME}"
echo "TEST_DIR=${TEST_DIR}"
[[ -d "${TEST_DIR}/images" ]] || { echo "ERROR: missing ${TEST_DIR}/images — run eval first" >&2; exit 1; }

bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"

echo "Done:"
echo "  ${TEST_DIR}/score_cd3_panck.csv"
echo "  ${TEST_DIR}/extended_metrics_per_tile.csv"
echo "  ${TEST_DIR}/extended_metrics_summary.json"
