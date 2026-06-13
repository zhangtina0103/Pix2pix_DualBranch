#!/bin/bash
# Build Orion Hoechst+CD3e+CD4 panel @512² (1480/500/500).
# Replaces Pan-CK with CD4 (moderate immune sparsity; Hoechst anchors FM).
#
#   sbatch bash_scripts/prepare_orion_cd3_cd4.sbatch
#SBATCH --job-name=orion_cd4_prep
#SBATCH --output=logs/orion_cd3_cd4_prepare_%j.out
#SBATCH --error=logs/orion_cd3_cd4_prepare_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

source "${REPO_ROOT}/bash_scripts/_orion_cd3_cd4_preamble.sh"

[[ -f "${ORION_SRC}/train_dataframe.csv" ]] || {
  echo "ERROR: missing ${ORION_SRC}/train_dataframe.csv" >&2
  exit 1
}

python scripts/prepare_orion_lite.py \
  --src "${ORION_SRC}" \
  --dst "${DATAROOT}" \
  --markers "${ORION_MARKERS}" \
  --n-train "${ORION_N_TRAIN}" \
  --n-val "${ORION_N_VAL}" \
  --n-test "${ORION_N_TEST}" \
  --tile-size "${ORION_TILE_SIZE}"

python3 -c "import json; m=json.load(open('${DATAROOT}/meta.json')); print(m)"
echo "CD3+CD4 panel ready: DATAROOT=${DATAROOT} markers=${ORION_MARKERS}"
echo "Pix2pix screen: sbatch bash_scripts/screen_orion_cd3_cd4.sbatch"
