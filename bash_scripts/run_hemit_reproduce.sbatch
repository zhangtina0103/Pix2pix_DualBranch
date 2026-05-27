#!/bin/bash
# HEMIT Dual-Branch: official README pipeline (prepare → train → test → post_process)
#
# First time only (if torch CUDA error): sbatch bash_scripts/setup_pix2pix_env.sbatch
#
# Submit from Pix2pix_DualBranch repo root:
#   mkdir -p logs
#   sbatch bash_scripts/run_hemit_reproduce.sbatch

#SBATCH --job-name=hemit_dual
#SBATCH --output=logs/hemit_dual_%j.out
#SBATCH --error=logs/hemit_dual_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate pix2pix
# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

mkdir -p logs datasets checkpoints results

HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
DATAROOT="${DATAROOT:-${REPO_ROOT}/datasets/hemit}"
MODE="${MODE:-all}"
export HEMIT_SRC DATAROOT MODE GPU_IDS=0

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Repo: ${REPO_ROOT}"
echo "HEMIT_SRC: ${HEMIT_SRC}"
echo "MODE: ${MODE}"
echo "CUDA_HOME: ${CUDA_HOME:-<unset>}"
echo "Started: $(date)"

if ! python -c "import torch; assert torch.cuda.is_available()"; then
  echo ""
  echo "ERROR: PyTorch cannot use CUDA (libcurand / libcublas missing)."
  echo "Run once:  sbatch bash_scripts/setup_pix2pix_env.sbatch"
  echo "Then:      sbatch bash_scripts/run_hemit_reproduce.sbatch"
  exit 1
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

bash scripts/run_hemit_reproduce.sh

echo "Finished: $(date)"
