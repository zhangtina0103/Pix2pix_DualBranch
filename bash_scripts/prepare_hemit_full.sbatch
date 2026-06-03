#!/bin/bash
# Build datasets/hemit from full HEMIT (trainA/B). Run once after copying full data to HEMIT_SRC.
#
#   export HEMIT_SRC=/home/zhangtin/HEMIT
#   export PREPARE_COPY=0
#   sbatch bash_scripts/prepare_hemit_full.sbatch
#SBATCH --job-name=hemit_prep
#SBATCH --output=logs/hemit_prepare_%j.out
#SBATCH --error=logs/hemit_prepare_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export PREPARE_COPY="${PREPARE_COPY:-0}"

echo "HEMIT_SRC=${HEMIT_SRC}"
echo "DATAROOT=${DATAROOT}"

for split in train val test; do
  n=$(ls "${HEMIT_SRC}/${split}/input" 2>/dev/null | wc -l)
  echo "  ${split}/input: ${n} files"
done

export MODEL=dualbranch
export MODE=prepare
bash scripts/run_hemit_all.sh

n=$(ls "${DATAROOT}/trainA" | wc -l)
echo "datasets/hemit trainA: ${n} (expect ~3717)"
[[ "${n}" -gt 1000 ]] || { echo "ERROR: still looks like a subset"; exit 1; }
