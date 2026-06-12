#!/bin/bash
# Build Orion immune panel dataroot: CD3e, CD8a, FOXP3 @512² (1480/500/500).
#
#   sbatch bash_scripts/prepare_orion_immune.sbatch
#SBATCH --job-name=orion_imm_prep
#SBATCH --output=logs/orion_immune_prepare_%j.out
#SBATCH --error=logs/orion_immune_prepare_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

source "${REPO_ROOT}/bash_scripts/_orion_immune_preamble.sh"

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
echo "Immune panel ready: DATAROOT=${DATAROOT} markers=${ORION_MARKERS}"
echo "GAN baselines: sbatch bash_scripts/train_orion_immune_gan_array.sbatch"
echo "cross-attn FM: sbatch bash_scripts/train_orion_immune_fm_cross_attn_scratch.sbatch"
