#!/bin/bash
# Download (optional) + build datasets/orion_lite from MIPHEI ORION tiles.
#
#   export ORION_SRC=/home/zhangtin/data/orion/ORIONCRC_dataset_tile_20x
#   sbatch bash_scripts/prepare_orion_lite.sbatch
#
# First-time download on cluster:
#   export ORION_DOWNLOAD=1
#   export ORION_DATA_DIR=/home/zhangtin/data/orion
#   sbatch bash_scripts/prepare_orion_lite.sbatch
#SBATCH --job-name=orion_prep
#SBATCH --output=logs/orion_prepare_%j.out
#SBATCH --error=logs/orion_prepare_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

export ORION_SRC="${ORION_SRC:-${REPO_ROOT}/data/orion/ORIONCRC_dataset_tile_20x}"
export DATAROOT="${DATAROOT:-./datasets/orion_lite}"
export ORION_N_TRAIN="${ORION_N_TRAIN:-1500}"
export ORION_TILE_SIZE="${ORION_TILE_SIZE:-512}"

if [[ "${ORION_DOWNLOAD:-0}" == "1" ]]; then
  python scripts/prepare_orion_lite.py \
    --download \
    --data-dir "${ORION_DATA_DIR:-${REPO_ROOT}/data/orion}" \
    --dst "${DATAROOT}" \
    --n-train "${ORION_N_TRAIN}" \
    --tile-size "${ORION_TILE_SIZE}"
else
  export MODE=prepare
  bash scripts/run_orion_all.sh
fi

python3 -c "import json; m=json.load(open('${DATAROOT}/meta.json')); print(m)"
echo "Orion-Lite ready at ${DATAROOT}"
