#!/usr/bin/env bash
# README-style pip venv (no conda). Run on a GPU node.
#
# Interactive:
#   cd ~/Pix2pix_DualBranch && bash scripts/setup_venv_engaging.sh
# SLURM:
#   sbatch bash_scripts/setup_venv_engaging.sbatch
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
VENV_DIR="${REPO_ROOT}/.venv-hemit"
CONSTRAINTS="${REPO_ROOT}/scripts/constraints_pix2pix.txt"

# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

find_python() {
  if [[ -n "${PYTHON_BIN:-}" ]] && [[ -x "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  local c
  for c in python3.10 \
    /home/zhangtin/miniforge3/bin/python3.10 \
    "$(command -v python3.10 2>/dev/null || true)" \
    "$(command -v python3 2>/dev/null || true)"; do
    [[ -n "${c}" && -x "${c}" ]] || continue
    if "${c}" -c 'import sys; assert sys.version_info[:2] == (3, 10)' 2>/dev/null; then
      echo "${c}"
      return
    fi
  done
  echo "ERROR: need Python 3.10 (set PYTHON_BIN=/path/to/python3.10)" >&2
  exit 1
}

PY_BOOT="$(find_python)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Creating venv with: ${PY_BOOT}"
echo "VENV_DIR=${VENV_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PY_BOOT}" -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

PIP="${VENV_DIR}/bin/pip"
PY="${VENV_DIR}/bin/python"

"${PIP}" install --upgrade pip setuptools wheel

"${PIP}" install --no-cache-dir --force-reinstall \
  -c "${CONSTRAINTS}" numpy==1.22.4 scipy==1.7.3

"${PIP}" install --no-cache-dir --force-reinstall \
  -c "${CONSTRAINTS}" \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118

verify_torch() {
  "${PY}" -c "
import numpy
import torch
assert numpy.__version__.startswith('1.22'), numpy.__version__
assert torch.cuda.is_available(), 'CUDA not available'
x = torch.zeros(1, device='cuda')
print('torch OK', torch.__version__, torch.version.cuda, 'numpy', numpy.__version__, x.device)
"
}

verify_torch

"${PIP}" install --no-cache-dir -c "${CONSTRAINTS}" \
  beautifulsoup4==4.12.3 \
  dominate==2.6.0 \
  einops==0.8.0 \
  matplotlib==3.5.1 \
  opencv-python-headless==4.5.5.62 \
  "Pillow>=8.4,<11" \
  Requests==2.32.3 \
  scikit-image==0.18.3 \
  tqdm==4.61.2

# timm pulls torch if not --no-deps (breaks cu118 wheels).
"${PIP}" install --no-cache-dir --no-deps -c "${CONSTRAINTS}" timm==0.4.12
"${PIP}" install --no-cache-dir --force-reinstall --no-deps -c "${CONSTRAINTS}" numpy==1.22.4

verify_torch
echo ""
echo "Done. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  source bash_scripts/_cuda.sh   # on GPU nodes"
echo "Train: sbatch bash_scripts/run_hemit_reproduce.sbatch"
