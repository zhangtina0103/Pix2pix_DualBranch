#!/bin/bash
# GAN paper baselines: test + post_process @80 (945 tiles).
#
#   sbatch --array=0-3 bash_scripts/eval_hemit_paper_gan_array.sbatch
#   0=pix2pix 1=cut 2=asp 3=cyclegan
#SBATCH --job-name=hemit_paper_gan
#SBATCH --output=logs/hemit_paper_gan_%A_%a.out
#SBATCH --error=logs/hemit_paper_gan_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

MODELS=(pix2pix cut asp cyclegan)
export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export TEST_EPOCH="${TEST_EPOCH:-80}"
export NUM_TEST="${NUM_TEST:-945}"
export GPU_IDS=0

source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"
if [[ "${MODEL}" == "cyclegan" ]]; then
  CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G_A.pth"
else
  CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
fi
echo "MODEL=${MODEL} TRAIN_NAME=${TRAIN_NAME} TEST_EPOCH=${TEST_EPOCH}"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT} — train first" >&2; exit 1; }

export MODE=test
bash scripts/run_hemit_all.sh
export MODE=metrics
bash scripts/run_hemit_all.sh
echo "Done → results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/"
