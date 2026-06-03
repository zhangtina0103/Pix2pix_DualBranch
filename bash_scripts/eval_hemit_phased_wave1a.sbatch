#!/bin/bash
# Wave 1a eval (2 jobs): pix2pix, pix2pixhd — test then metrics
#SBATCH --job-name=hemit_w1a_ev
#SBATCH --output=logs/hemit_w1a_eval_%A_%a.out
#SBATCH --error=logs/hemit_w1a_eval_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

MODELS=(pix2pix pix2pixhd)
export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export TEST_EPOCH="${TEST_EPOCH:-80}"
export NUM_TEST=945
export GPU_IDS=0

source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"
CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
echo "Wave1a eval: MODEL=${MODEL} TRAIN_NAME=${TRAIN_NAME}"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}"; exit 1; }

export MODE=test
bash scripts/run_hemit_all.sh
export MODE=metrics
bash scripts/run_hemit_all.sh
