#!/bin/bash
#SBATCH --job-name=hemit
#SBATCH --output=logs/hemit_%j.out
#SBATCH --error=logs/hemit_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
# pix2pix_cuda: dualbranch + pix2pix | virtual-staining: cut/asp/cyclegan/fm if extra deps installed
CONDA_ENV="${CONDA_ENV:-pix2pix_cuda}"
conda activate "${CONDA_ENV}"
# shellcheck source=/dev/null
[[ "${CONDA_ENV}" == pix2pix* ]] && source "${REPO_ROOT}/bash_scripts/_cuda.sh" || true

export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export VS_DATA_ROOT="${VS_DATA_ROOT:-${HEMIT_SRC}}"
export MODEL="${MODEL:-dualbranch}"
export MODE="${MODE:-all}"
export GPU_IDS=0

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "MODEL=${MODEL} MODE=${MODE} CONDA=${CONDA_ENV}"
echo "Started: $(date)"

python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || true
bash scripts/run_hemit_all.sh
echo "Finished: $(date)"
