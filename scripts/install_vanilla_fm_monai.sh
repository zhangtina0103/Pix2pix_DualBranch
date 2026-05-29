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

echo ""
echo "Done. Train with: sbatch bash_scripts/train_hemit_vanilla_fm.sbatch"
echo "If import still fails: git pull (mentor_flow_net.py generative fallback)"
