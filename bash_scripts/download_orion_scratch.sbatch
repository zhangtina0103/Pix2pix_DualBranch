#!/bin/bash
# Download MIPHEI ORION zip (~118 GiB) to scratch. Run once.
#
#   sbatch bash_scripts/download_orion_scratch.sbatch
#
# Resume: re-submit same command (aria2c --continue=true).
# Monitor: tail -f logs/orion_download_*.out
#SBATCH --job-name=orion_dl
#SBATCH --output=logs/orion_download_%j.out
#SBATCH --error=logs/orion_download_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/orion_scratch_env.sh"
mkdir -p "${ORION_SCRATCH_ROOT}"
cd "${ORION_SCRATCH_ROOT}"

echo "ORION_SCRATCH_ROOT=${ORION_SCRATCH_ROOT}"
echo "Target zip: ${ORION_ZIP}"
df -h "${ORION_SCRATCH_ROOT}"

if [[ -f "${ORION_ZIP}" ]]; then
  SZ=$(stat -c%s "${ORION_ZIP}" 2>/dev/null || stat -f%z "${ORION_ZIP}")
  if [[ "${SZ}" -ge 125000000000 ]]; then
    echo "Zip already complete ($(numfmt --to=iec-i --suffix=B "${SZ}" 2>/dev/null || echo "${SZ} bytes"))."
    exit 0
  fi
  echo "Partial zip found ($(numfmt --to=iec-i --suffix=B "${SZ}" 2>/dev/null || echo "${SZ} bytes")) — resuming."
fi

aria2c --check-certificate=false --file-allocation=none \
  -x 16 -s 16 -k 10M --continue=true \
  -o ORIONCRC_dataset_tile_20x.zip \
  "${ZENODO_ORION_URL}"

SZ=$(stat -c%s "${ORION_ZIP}" 2>/dev/null || stat -f%z "${ORION_ZIP}")
echo "Download finished: ${SZ} bytes"
[[ "${SZ}" -ge 125000000000 ]] || { echo "ERROR: zip still too small; download incomplete" >&2; exit 1; }
echo "Next: sbatch bash_scripts/extract_orion_scratch.sbatch"
