#!/bin/bash
# Test then post_process metrics (two steps; do not use MODE=test|metrics).
#
#   export HEMIT_SRC=/home/zhangtin/HEMIT
#   MODEL=pix2pix TEST_EPOCH=80 sbatch bash_scripts/eval_hemit_native_full.sbatch
#SBATCH --job-name=hemit_ev
#SBATCH --output=logs/hemit_eval_%x_%j.out
#SBATCH --error=logs/hemit_eval_%x_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL="${MODEL:?Set MODEL=pix2pix|pix2pixhd|cut|asp|cyclegan before sbatch}"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export TEST_EPOCH="${TEST_EPOCH:-80}"
export NUM_TEST="${NUM_TEST:-945}"
export GPU_IDS=0

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"
CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
echo "MODEL=${MODEL} TRAIN_NAME=${TRAIN_NAME} TEST_EPOCH=${TEST_EPOCH}"
echo "checkpoint: ${CKPT}"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing checkpoint — train first"; exit 1; }

echo "===== test ====="
export MODE=test
bash scripts/run_hemit_all.sh

echo "===== metrics ====="
export MODE=metrics
bash scripts/run_hemit_all.sh

echo "Done. See results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/score.csv"
