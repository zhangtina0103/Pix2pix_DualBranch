#!/usr/bin/env bash
# GPU PyTorch 2.0 + cu118 for pix2pix env. Run on a GPU node only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate pix2pix
# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

PY="${CONDA_PREFIX}/bin/python"
PIP="${CONDA_PREFIX}/bin/pip"
SITE="${CONDA_PREFIX}/lib/python3.10/site-packages"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CONDA_PREFIX=${CONDA_PREFIX}"

# Remove mixed conda + pip torch (causes torch._C attribute errors).
conda remove -y pytorch torchvision torchaudio libtorch pytorch-cuda 2>/dev/null || true
"${PIP}" uninstall -y torch torchvision torchaudio 2>/dev/null || true
rm -rf "${SITE}"/torch "${SITE}"/torchvision "${SITE}"/torch-*.dist-info \
  "${SITE}"/torchvision-*.dist-info "${SITE}"/functorch 2>/dev/null || true

conda install -y setuptools wheel pip
"${PIP}" install --upgrade pip setuptools wheel

# Phase 1: PyTorch only — verify before anything else touches the env.
"${PIP}" install --no-cache-dir --force-reinstall \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118

"${PY}" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
x = torch.zeros(1, device='cuda')
print('torch OK', torch.__version__, torch.version.cuda, x.device)
"

# Phase 2: project deps (no torch; wandb optional for --use_wandb only).
"${PIP}" install --no-cache-dir \
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
  tqdm==4.61.2

# Phase 3: re-check torch was not broken by later installs.
"${PY}" -c "
import torch
assert torch.cuda.is_available()
print('final OK', torch.__version__, torch.version.cuda)
"

echo "Done. HEMIT reproduce uses display_id 0 (no visdom/wandb)."
echo "Optional logging: pip install 'wandb==0.12.7' && train.py --use_wandb"
