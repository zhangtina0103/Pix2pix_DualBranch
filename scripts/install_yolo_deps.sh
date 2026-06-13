#!/bin/bash
# YOLO deps for pix2pix_cuda without upgrading numpy to 2.x (breaks torch/matplotlib).
#
# Usage (interactive):
#   conda activate pix2pix_cuda
#   bash scripts/install_yolo_deps.sh
#
set -euo pipefail

NUMPY_PIN="${NUMPY_PIN:-1.22.4}"

echo "[yolo-deps] pin numpy==${NUMPY_PIN}"
pip install -q "numpy==${NUMPY_PIN}"

echo "[yolo-deps] scikit-image + tifffile (numpy 1.x stack)"
pip install -q "scikit-image>=0.19,<0.25" "tifffile>=2023.1"

echo "[yolo-deps] ultralytics (no-deps — do not upgrade numpy/torch)"
pip install -q "ultralytics>=8.0" --no-deps
pip install -q \
  "opencv-python-headless>=4.6" \
  "pyyaml" \
  "tqdm" \
  "requests" \
  "psutil" \
  "py-cpuinfo" \
  "pandas" \
  "matplotlib" \
  "seaborn" \
  "pillow" \
  "scipy"

python - <<'PY'
import numpy as np
import torch
import skimage
from ultralytics import YOLO
print(f"OK numpy={np.__version__} torch={torch.__version__} skimage={skimage.__version__} ultralytics YOLO import")
PY
