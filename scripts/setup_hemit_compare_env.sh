#!/usr/bin/env bash
# Second conda env for hemit/ models (CUT, ASP, CycleGAN, FM) — do NOT touch pix2pix_cuda.
#
# Run on a GPU node (or login for create only; verify CUDA on GPU):
#   bash scripts/setup_hemit_compare_env.sh
#
# Then submit:
#   conda activate hemit_compare
#   MODEL=cut MARKER=CD3 MODE=train sbatch bash_scripts/run_hemit_all.sbatch
#   # or: sbatch bash_scripts/train_hemit_gan_array.sbatch
set -euo pipefail

ENV_NAME="${ENV_NAME:-hemit_compare}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/scripts/constraints_pix2pix.txt"

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh

if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "Creating ${ENV_NAME} (python 3.10)..."
  conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"

# shellcheck source=/dev/null
[[ -f "${REPO_ROOT}/bash_scripts/_cuda.sh" ]] && source "${REPO_ROOT}/bash_scripts/_cuda.sh"

PIP="${CONDA_PREFIX}/bin/pip"
PY="${CONDA_PREFIX}/bin/python"

"${PIP}" install --upgrade pip setuptools wheel

# --- core (same pins as pix2pix_cuda) ---
"${PIP}" install --no-cache-dir --force-reinstall \
  -c "${CONSTRAINTS}" numpy==1.22.4 scipy==1.7.3

"${PIP}" install --no-cache-dir --force-reinstall \
  -c "${CONSTRAINTS}" \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118

verify_core() {
  "${PY}" -c "
import numpy, scipy, torch
assert numpy.__version__.startswith('1.22'), numpy.__version__
assert torch.cuda.is_available(), 'CUDA not available on this node'
print('OK core', 'numpy', numpy.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))
"
}

verify_core

# --- hemit/ comparison deps (one group; no torch upgrade) ---
for pkg in \
  "matplotlib==3.5.1" \
  "opencv-python-headless==4.5.5.62" \
  "Pillow>=8.4,<11" \
  "tqdm==4.61.2" \
  "einops==0.8.0" \
  "huggingface_hub>=0.20" \
  "torchmetrics>=1.0,<2" \
  "lpips>=0.1.4"; do
  echo "==> pip install ${pkg}"
  "${PIP}" install --no-cache-dir -c "${CONSTRAINTS}" "${pkg}"
done

# lightning + monai (FM); install without deps then missing pieces manually if needed
echo "==> pip install lightning monai"
"${PIP}" install --no-cache-dir -c "${CONSTRAINTS}" "lightning>=2.0,<3" "monai>=1.3,<2"

conda install -y -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12" || true

verify_core

"${PY}" -c "
import lightning
import monai
import lpips
import skimage
print('OK hemit', 'lightning', lightning.__version__)
"

echo ""
echo "Done. Use:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${REPO_ROOT}"
echo "  MODEL=cut MARKER=CD3 MODE=train sbatch bash_scripts/run_hemit_all.sbatch"
echo "  # or: sbatch bash_scripts/train_hemit_gan_array.sbatch"
