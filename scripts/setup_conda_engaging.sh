#!/usr/bin/env bash
# Fix pix2pix conda env on Engaging: PyTorch + CUDA libs (run on a GPU compute node).
#
#   srun --gres=gpu:1 --pty bash
#   cd ~/Pix2pix_DualBranch && bash scripts/setup_conda_engaging.sh
#
# Then submit: sbatch bash_scripts/run_hemit_reproduce.sbatch

set -euo pipefail

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
ENV_NAME="${CONDA_ENV:-pix2pix}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi
module purge 2>/dev/null || true
module load cuda/12.2 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$ENV_NAME"

echo "Installing PyTorch with CUDA (conda, not pip-only)..."
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Installing other deps..."
pip install -r requirements.txt

python -c "import torch; print('OK:', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "Done. Submit: sbatch bash_scripts/run_hemit_reproduce.sbatch"
