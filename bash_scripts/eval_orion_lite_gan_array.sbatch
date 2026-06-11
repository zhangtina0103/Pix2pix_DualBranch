#!/bin/bash
# Orion-Lite GAN baseline eval @80.
#   sbatch bash_scripts/eval_orion_lite_gan_array.sbatch
#SBATCH --job-name=orion_gan_ev
#SBATCH --output=logs/orion_gan_eval_%A_%a.out
#SBATCH --error=logs/orion_gan_eval_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

MODELS=(pix2pix cut asp cyclegan)
export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export DATAROOT="${DATAROOT:-./datasets/orion_lite}"
export TEST_EPOCH="${TEST_EPOCH:-80}"
export GPU_IDS=0

source "${REPO_ROOT}/scripts/orion_model_profiles.sh"
CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
echo "MODEL=${MODEL} TRAIN_NAME=${TRAIN_NAME} NUM_TEST=${NUM_TEST}"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}" >&2; exit 1; }

export MODE=test
bash scripts/run_orion_native.sh
export MODE=metrics
bash scripts/run_orion_native.sh

echo "score.csv → results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/score.csv"
