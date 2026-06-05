#!/bin/bash
# DiffVS two-stage train on HEMIT @ 512² (uses ../DiffVS upstream code).
#
#   export HEMIT_SRC=/home/zhangtin/HEMIT
#   export DIFFVS_ROOT=/home/zhangtin/DiffVS
#   sbatch bash_scripts/train_hemit_diffvs.sbatch
#SBATCH --job-name=hemit_diffvs
#SBATCH --output=logs/hemit_diffvs_train_%j.out
#SBATCH --error=logs/hemit_diffvs_train_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL=diffvs
export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export IMAGE_SIZE="${IMAGE_SIZE:-512}"
export DIFFVS_TRAIN_BS="${DIFFVS_TRAIN_BS:-4}"
export DIFFVS_ROOT="${DIFFVS_ROOT:-${REPO_ROOT}/../DiffVS}"

bash scripts/run_hemit_all.sh
