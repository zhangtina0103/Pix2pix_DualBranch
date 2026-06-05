#!/bin/bash
# D-VST finetune on HEMIT @ 512² (separate conda env recommended: D_VST).
#
#   export HEMIT_SRC=/home/zhangtin/HEMIT
#   export DVST_ROOT=/home/zhangtin/D-VST
#   export CONDA_ENV=D_VST
#   sbatch bash_scripts/train_hemit_dvst.sbatch
#SBATCH --job-name=hemit_dvst
#SBATCH --output=logs/hemit_dvst_train_%j.out
#SBATCH --error=logs/hemit_dvst_train_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-D_VST}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh" 2>/dev/null || true

export MODEL=dvst
export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export IMAGE_SIZE="${IMAGE_SIZE:-512}"
export DVST_ROOT="${DVST_ROOT:-${REPO_ROOT}/../D-VST}"
export DVST_TRAIN_BS="${DVST_TRAIN_BS:-2}"

bash scripts/run_hemit_all.sh
