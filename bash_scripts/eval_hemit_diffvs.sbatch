#!/bin/bash
# DiffVS infer + metrics on HEMIT test (945 patches).
#
#   export DIFFVS_STAGE2_EPOCH=5
#   sbatch bash_scripts/eval_hemit_diffvs.sbatch
#SBATCH --job-name=hemit_diffvs_ev
#SBATCH --output=logs/hemit_diffvs_eval_%j.out
#SBATCH --error=logs/hemit_diffvs_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results/diffusion

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL=diffvs
export MODE="test|metrics"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export IMAGE_SIZE="${IMAGE_SIZE:-512}"
export DIFFVS_ROOT="${DIFFVS_ROOT:-${REPO_ROOT}/../DiffVS}"

bash scripts/run_hemit_all.sh
