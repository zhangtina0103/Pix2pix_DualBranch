#!/bin/bash
# Train native HEMIT baselines on FULL data (one model per array task).
#
#   export HEMIT_SRC=/home/zhangtin/HEMIT_full
#   sbatch bash_scripts/prepare_hemit_full.sbatch
#   sbatch bash_scripts/run_hemit_native_train_array.sbatch
#
# Models: 0=dualbranch 1=pix2pix 2=pix2pixhd 3=cut 4=asp 5=cyclegan
#SBATCH --job-name=hemit_tr
#SBATCH --output=logs/hemit_train_%A_%a.out
#SBATCH --error=logs/hemit_train_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

MODELS=(dualbranch pix2pix pix2pixhd cut asp cyclegan)
export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT_full}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export GPU_IDS=0

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL=${MODEL}"
ls "${DATAROOT}/trainA" | wc -l
bash scripts/run_hemit_all.sh
