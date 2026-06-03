#!/bin/bash
# Wave 1 train (4 jobs): pix2pix, pix2pixhd, CUT, ASP
#SBATCH --job-name=hemit_w1_tr
#SBATCH --output=logs/hemit_w1_train_%A_%a.out
#SBATCH --error=logs/hemit_w1_train_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-3
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

MODELS=(pix2pix pix2pixhd cut asp)
export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export GPU_IDS=0

echo "Wave1 train task ${SLURM_ARRAY_TASK_ID}: MODEL=${MODEL}"
bash scripts/run_hemit_all.sh
