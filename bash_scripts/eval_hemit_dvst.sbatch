#!/bin/bash
# D-VST eval on HEMIT test (uses eval.py + post_process.py).
#
#   export DVST_EVAL_CKPT=/path/to/HE2mIHC.ckpt
#   sbatch bash_scripts/eval_hemit_dvst.sbatch
#SBATCH --job-name=hemit_dvst_ev
#SBATCH --output=logs/hemit_dvst_eval_%j.out
#SBATCH --error=logs/hemit_dvst_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results/diffusion

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-D_VST}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL=dvst
export MODE="test|metrics"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export IMAGE_SIZE="${IMAGE_SIZE:-512}"
export DVST_ROOT="${DVST_ROOT:-${REPO_ROOT}/../D-VST}"
export DVST_EVAL_CKPT="${DVST_EVAL_CKPT:-${DVST_ROOT}/weights/dvst_pretrained/HE2mIHC.ckpt}"

bash scripts/run_hemit_all.sh
