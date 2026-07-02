#!/bin/bash
# Prepare HEMIT multi-input dataroots (he / dapi / he_dapi → CD3+panCK).
#
#   sbatch bash_scripts/prepare_hemit_multiinput.sbatch
#   # or locally:
#   python scripts/prepare_hemit_multiinput.py --src ./datasets/hemit
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

export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT_HEMIT="${DATAROOT_HEMIT:-./datasets/hemit}"
export HEMIT_MULTI_ROOT="${HEMIT_MULTI_ROOT:-./datasets/hemit_multi}"

# Ensure base pix2pix dataroot exists
if [[ ! -d "${DATAROOT_HEMIT}/trainA" ]]; then
  echo "==> Building ${DATAROOT_HEMIT} from ${HEMIT_SRC}"
  python scripts/prepare_hemit_data.py --src "${HEMIT_SRC}" --dst "${DATAROOT_HEMIT}"
fi

python scripts/prepare_hemit_multiinput.py \
  --src "${DATAROOT_HEMIT}" \
  --dst-base "${HEMIT_MULTI_ROOT}"

echo "Done. Dataroots under ${HEMIT_MULTI_ROOT}/{he,dapi,he_dapi}"
