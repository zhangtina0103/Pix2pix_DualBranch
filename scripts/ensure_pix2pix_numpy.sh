#!/usr/bin/env bash
# pix2pix_cuda requires numpy 1.x (torch/matplotlib/timm ABI). YOLO/ultralytics
# can silently upgrade to numpy 2.x — restore the pin before train/test jobs.
ensure_pix2pix_numpy() {
  local pin="${PIX2PIX_NUMPY_PIN:-1.22.4}"
  if ! python -c "import numpy as np; assert np.__version__.startswith('1.'), np.__version__" 2>/dev/null; then
    echo "[ensure_pix2pix_numpy] numpy 2.x detected — pinning to ${pin}"
    pip install -q "numpy==${pin}"
  fi
  python -c "
import numpy as np
import matplotlib.pyplot
import torch, torchvision
print(f'[ensure_pix2pix_numpy] OK numpy={np.__version__} torch={torch.__version__}')
" || {
    echo "[ensure_pix2pix_numpy] import check failed after pin — run manually:" >&2
    echo "  pip install numpy==${pin} 'matplotlib>=3.5,<3.8'" >&2
    return 1
  }
}
