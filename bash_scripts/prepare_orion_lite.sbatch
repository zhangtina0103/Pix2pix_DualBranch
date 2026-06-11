#!/bin/bash
# Build Orion-Lite on scratch (requires extracted ORION tiles).
#
#   sbatch bash_scripts/prepare_orion_lite.sbatch
#
# Full pipeline:
#   sbatch bash_scripts/download_orion_scratch.sbatch
#   sbatch bash_scripts/extract_orion_scratch.sbatch
#   sbatch bash_scripts/prepare_orion_lite.sbatch
#SBATCH --job-name=orion_prep
#SBATCH --output=logs/orion_prepare_%j.out
#SBATCH --error=logs/orion_prepare_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/orion_scratch_env.sh"
export ORION_N_TRAIN="${ORION_N_TRAIN:-1500}"
export ORION_TILE_SIZE="${ORION_TILE_SIZE:-512}"

[[ -f "${ORION_SRC}/train_dataframe.csv" ]] || {
  echo "ERROR: missing ${ORION_SRC}/train_dataframe.csv" >&2
  echo "Run: sbatch bash_scripts/download_orion_scratch.sbatch" >&2
  echo "     sbatch bash_scripts/extract_orion_scratch.sbatch" >&2
  exit 1
}

python scripts/prepare_orion_lite.py \
  --src "${ORION_SRC}" \
  --dst "${DATAROOT}" \
  --n-train "${ORION_N_TRAIN}" \
  --tile-size "${ORION_TILE_SIZE}"

python3 -c "import json; m=json.load(open('${DATAROOT}/meta.json')); print(m)"
echo "Orion-Lite ready: DATAROOT=${DATAROOT}"
echo "Train: sbatch bash_scripts/train_orion_lite_fm_cross_attn_scratch.sbatch"
