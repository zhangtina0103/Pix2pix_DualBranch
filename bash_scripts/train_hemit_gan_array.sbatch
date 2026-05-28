#!/bin/bash
# 9 jobs: cyclegan, cut, asp × DAPI, panCK, CD3
#   sbatch bash_scripts/train_hemit_gan_array.sbatch
#SBATCH --job-name=hemit_gan
#SBATCH --output=logs/hemit_gan_%A_%a.out
#SBATCH --error=logs/hemit_gan_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-hemit_compare}"

MODELS=(cyclegan cyclegan cyclegan cut cut cut asp asp asp)
MARKERS=(DAPI panCK CD3 DAPI panCK CD3 DAPI panCK CD3)
export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export MARKER="${MARKERS[$SLURM_ARRAY_TASK_ID]}"
export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export VS_DATA_ROOT="${VS_DATA_ROOT:-${HEMIT_SRC}}"

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL=${MODEL} MARKER=${MARKER}"
bash scripts/run_hemit_all.sh
