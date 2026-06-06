#!/usr/bin/env bash
# Install FM backbone deps WITHOUT upgrading numpy/torch (pix2pix_cuda pins).
#
# monai>=1.4 needs numpy>=1.24 → conflicts with constraints_pix2pix.txt.
# Use monai-generative (DiffusionModelUNet) + existing monai 1.3.x instead.
#
# Usage (login or GPU node):
#   bash scripts/install_vanilla_fm_monai.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"

echo "==> numpy / torch (unchanged)"
"${PY}" -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"

echo "==> monai (keep 1.3.x if present; do not upgrade to 1.4+ here)"
if ! "${PY}" -c "import monai" 2>/dev/null; then
  pip install --no-cache-dir -c "${ROOT}/scripts/constraints_pix2pix.txt" "monai>=1.3,<1.4"
fi
"${PY}" -c "import monai; print('monai', monai.__version__)"

echo "==> monai-generative (DiffusionModelUNet)"
pip install --no-cache-dir "monai-generative>=0.2.3,<0.3"

echo "==> verify DiffusionModelUNet import"
"${PY}" -c "
from generative.networks.nets import DiffusionModelUNet
from monai.losses import PerceptualLoss
print('OK: generative DiffusionModelUNet + monai PerceptualLoss')
"

echo "==> optional: LPIPS perceptual (if monai PerceptualLoss missing)"
pip install --no-cache-dir "lpips>=0.1.4" || true

echo "==> optional: xformers (MONAI use_flash_attention; must match torch/CUDA)"
TORCH_VER="$("${PY}" -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")"
case "${TORCH_VER}" in
  2.0) XFORMERS_VER="0.0.22" ;;
  2.1) XFORMERS_VER="0.0.23.post1" ;;
  2.2) XFORMERS_VER="0.0.25.post1" ;;
  *)
    echo "WARN: no pinned xformers wheel for torch ${TORCH_VER}; skip or pick version manually" >&2
    XFORMERS_VER=""
    ;;
esac
if [[ -n "${XFORMERS_VER}" ]]; then
  if ! "${PY}" -c "import xformers" 2>/dev/null; then
    pip install --no-cache-dir "xformers==${XFORMERS_VER}" || \
      echo "WARN: xformers install failed — MONAI falls back to use_flash_attention=False" >&2
  fi
  "${PY}" -c "import xformers; print('xformers', xformers.__version__)" 2>/dev/null || true
fi

echo ""
echo "Done. Train with: sbatch bash_scripts/train_hemit_vanilla_fm_monai512.sbatch"
echo "Log should show: FM perceptual: backend=monai|lpips|vgg ..."
