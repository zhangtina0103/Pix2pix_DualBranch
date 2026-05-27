#!/bin/bash
# One-time: fix pix2pix env — PyTorch with CUDA (pip-only torch misses libcurand/cublas).
#
# Submit once from repo root, wait for success, then:
#   sbatch bash_scripts/run_hemit_reproduce.sbatch
#
#SBATCH --job-name=setup_pix2pix
#SBATCH --output=logs/setup_pix2pix_%j.out
#SBATCH --error=logs/setup_pix2pix_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate pix2pix
# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"

# Remove broken pip-only torch (causes libcurand.so.10 / libcublas errors)
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo "Installing PyTorch + CUDA via conda..."
conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

echo "Installing repo dependencies (skip pinned torch lines)..."
pip install \
  beautifulsoup4 dominate einops matplotlib numpy opencv_python \
  Pillow Requests scikit_image scipy timm tqdm visdom wandb

python -c "import torch; assert torch.cuda.is_available(), 'CUDA still not visible'; print('OK', torch.__version__, torch.version.cuda)"

echo "Finished: $(date)"
echo "Next: sbatch bash_scripts/run_hemit_reproduce.sbatch"
