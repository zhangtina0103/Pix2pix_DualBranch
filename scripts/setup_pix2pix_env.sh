#!/usr/bin/env bash
# GPU PyTorch 2.0 + cu118 for pix2pix env. Run on a GPU node only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate pix2pix
# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CONDA_PREFIX=${CONDA_PREFIX}"

conda remove -y pytorch torchvision torchaudio libtorch 2>/dev/null || true
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

conda install -y setuptools wheel pip

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

pip install \
  "numpy>=1.19,<2" \
  beautifulsoup4==4.12.3 \
  dominate==2.6.0 \
  einops==0.8.0 \
  matplotlib==3.5.1 \
  opencv-python-headless==4.5.5.62 \
  "Pillow>=8.4,<11" \
  Requests==2.32.3 \
  scikit-image==0.18.3 \
  scipy==1.7.3 \
  timm==0.4.12 \
  tqdm==4.61.2 \
  wandb==0.12.7

python -c "import torch; assert torch.cuda.is_available(); print('OK', torch.__version__, torch.version.cuda)"
