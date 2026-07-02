#!/bin/bash
# Prepare HEMIT multi-input dataroots on ORCD scratch (not home).
#
#   sbatch bash_scripts/prepare_hemit_multiinput.sbatch
#
# Scratch layout:
#   ~/orcd/scratch/hemit/datasets/hemit/       — base pix2pix (symlinks to HEMIT_SRC)
#   ~/orcd/scratch/hemit/datasets/hemit_multi/ — he | dapi | he_dapi
#SBATCH --job-name=hemit_multi_prep
#SBATCH --output=logs/hemit_multiinput_prep_%j.out
#SBATCH --error=logs/hemit_multiinput_prep_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hemit_scratch_env.sh"
hemit_apply_scratch_env

# Base pix2pix dataroot on scratch (symlinks — minimal extra disk vs HEMIT_SRC on home)
if [[ ! -d "${DATAROOT_HEMIT}/trainA" ]]; then
  echo "==> Building ${DATAROOT_HEMIT} from ${HEMIT_SRC} (symlinks)"
  python scripts/prepare_hemit_data.py --src "${HEMIT_SRC}" --dst "${DATAROOT_HEMIT}"
fi

python scripts/prepare_hemit_multiinput.py \
  --src "${DATAROOT_HEMIT}" \
  --dst-base "${HEMIT_MULTI_ROOT}" \
  --variants "${HEMIT_MULTI_VARIANTS:-he,dapi,he_dapi}"

echo "Done."
echo "  base:   ${DATAROOT_HEMIT}"
echo "  multi:  ${HEMIT_MULTI_ROOT}/{he,dapi,he_dapi}"
du -sh "${HEMIT_MULTI_ROOT}"/* 2>/dev/null || true
