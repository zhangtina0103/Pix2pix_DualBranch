#!/bin/bash
# 3 jobs: vanilla FM × DAPI, panCK, CD3  (+ separate sbatch for fm_plus)
#SBATCH --job-name=hemit_fm
#SBATCH --output=logs/hemit_fm_%A_%a.out
#SBATCH --error=logs/hemit_fm_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"

MARKERS=(DAPI panCK CD3)
export MODEL=fm
export MARKER="${MARKERS[$SLURM_ARRAY_TASK_ID]}"
export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"

bash scripts/run_hemit_all.sh
