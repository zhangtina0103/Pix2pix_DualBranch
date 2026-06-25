#!/bin/bash
# Prepare CaMSC 5-fold dataroots on scratch.
#
#   # After rsync zip + unzip to ~/orcd/scratch/camsc/20260504/
#   sbatch bash_scripts/prepare_camsc_bf.sbatch
#
#   # Then train all folds:
#   sbatch bash_scripts/train_camsc_bf_kfold_array.sbatch
#SBATCH --job-name=camsc_prep
#SBATCH --output=logs/camsc_prepare_%j.out
#SBATCH --error=logs/camsc_prepare_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/camsc_scratch_env.sh"

[[ -d "${CAMSC_SRC}" ]] || {
  echo "ERROR: missing ${CAMSC_SRC}" >&2
  echo "  rsync zip to ~/orcd/scratch/camsc/ && unzip" >&2
  exit 1
}

python scripts/prepare_camsc_bf.py \
  --src "${CAMSC_SRC}" \
  --auto-discover \
  --k-folds "${CAMSC_KFOLDS}" \
  --dst "${CAMSC_KFOLD_ROOT}" \
  --tile-size "${CAMSC_TILE_SIZE:-512}" \
  --seed "${CAMSC_SEED:-42}"

cat "${CAMSC_KFOLD_ROOT}/folds.json"
echo "Ready: ${CAMSC_KFOLD_ROOT}/fold{0..$((CAMSC_KFOLDS - 1))}"
echo "Train: sbatch bash_scripts/train_camsc_bf_kfold_array.sbatch"
