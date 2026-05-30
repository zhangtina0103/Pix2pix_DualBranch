#!/bin/bash
# Test + metrics: hemit_vanilla_fm_joint_film @ epoch 100 (100 tiles, 25-step Heun).
#
#   sbatch bash_scripts/test_hemit_vanilla_fm_joint_film.sbatch
#
# Metrics only (after test finished):
#   MODE=metrics sbatch bash_scripts/test_hemit_vanilla_fm_joint_film.sbatch
#
# Other epoch:
#   TEST_EPOCH=80 sbatch bash_scripts/test_hemit_vanilla_fm_joint_film.sbatch
#
# Do NOT use: sbatch --export=ALL ...
#SBATCH --job-name=hemit_vfm_film_t
#SBATCH --output=logs/hemit_vfm_joint_film_test_%j.out
#SBATCH --error=logs/hemit_vfm_joint_film_test_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL=vanilla_fm
export MODE="${MODE:-test|metrics}"
export GPU_IDS=0
export DISPLAY_ID="${DISPLAY_ID:--1}"

export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"

export TEST_EPOCH="${TEST_EPOCH:-100}"
export NUM_TEST="${NUM_TEST:-100}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_joint_film_env
export VANILLA_FM_ENV_LOCKED=1

CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
[[ -f "${CKPT}" ]] || {
  echo "ERROR: missing ${CKPT} (wait for train to save epoch ${TEST_EPOCH})" >&2
  exit 1
}

vanilla_fm_print_train_env
echo "MODE=${MODE} TEST_EPOCH=${TEST_EPOCH} NUM_TEST=${NUM_TEST}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"

bash "${REPO_ROOT}/scripts/run_hemit_all.sh"

echo "Finished: $(date)"
echo "Fakes: results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/"
echo "Metrics: results/${TRAIN_NAME}/test_${TEST_EPOCH}/score.csv"
