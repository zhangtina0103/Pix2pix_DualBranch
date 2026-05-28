#!/bin/bash
# 3 jobs: joint 3-channel CUT, ASP, CycleGAN (same pipeline as pix2pix/dualbranch)
#   sbatch bash_scripts/train_hemit_gan_array.sbatch
# CycleGAN uses batch_size=1 by default (4 nets @ 1024²). Still OOM: LAMBDA_IDENTITY=0
#SBATCH --job-name=hemit_gan
#SBATCH --output=logs/hemit_gan_%A_%a.out
#SBATCH --error=logs/hemit_gan_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

MODELS=(cut asp cyclegan)
export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL=${MODEL} (joint 3ch)"
bash scripts/run_hemit_all.sh
