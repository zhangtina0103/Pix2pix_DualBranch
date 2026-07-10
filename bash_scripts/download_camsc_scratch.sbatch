#!/bin/bash
# Resumable CaMSC zip download to ORCD scratch (~8.2 GiB from Dropbox).
#
# 1. In Dropbox: Share → Copy link → change ?dl=0 to ?dl=1
# 2. On Engaging:
#      export CAMSC_DOWNLOAD_URL='https://www.dropbox.com/s/.../file.zip?dl=1'
#      sbatch bash_scripts/download_camsc_scratch.sbatch
#
# Re-submit until complete (aria2c --continue=true resumes partial files):
#   while ! test $(stat -c%s ~/orcd/scratch/camsc/camsc_20260709.zip 2>/dev/null || echo 0) \
#     -ge 8500000000; do sbatch bash_scripts/download_camsc_scratch.sbatch; sleep 60; done
#
# Monitor: tail -f logs/camsc_download_*.out
#SBATCH --job-name=camsc_dl
#SBATCH --output=logs/camsc_download_%j.out
#SBATCH --error=logs/camsc_download_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/camsc_scratch_env.sh"
mkdir -p "${CAMSC_SCRATCH_ROOT}" || {
  echo "ERROR: cannot create ${CAMSC_SCRATCH_ROOT}" >&2
  exit 1
}
cd "${CAMSC_SCRATCH_ROOT}"

ZIP_NAME="$(basename "${CAMSC_ZIP}")"
MIN_BYTES="${CAMSC_ZIP_MIN_BYTES:-8500000000}"

echo "CAMSC_SCRATCH_ROOT=${CAMSC_SCRATCH_ROOT}"
echo "Target zip: ${CAMSC_ZIP}"
echo "Min bytes:  ${MIN_BYTES}"
df -h "${CAMSC_SCRATCH_ROOT}"

if [[ -z "${CAMSC_DOWNLOAD_URL:-}" ]]; then
  echo "ERROR: set CAMSC_DOWNLOAD_URL to a Dropbox direct link (?dl=1)" >&2
  echo "  export CAMSC_DOWNLOAD_URL='https://www.dropbox.com/s/.../file.zip?dl=1'" >&2
  exit 1
fi

if [[ -f "${CAMSC_ZIP}" ]]; then
  SZ=$(stat -c%s "${CAMSC_ZIP}" 2>/dev/null || stat -f%z "${CAMSC_ZIP}")
  if [[ "${SZ}" -ge "${MIN_BYTES}" ]]; then
    echo "Zip already complete ($(numfmt --to=iec-i --suffix=B "${SZ}" 2>/dev/null || echo "${SZ} bytes"))."
    echo "Next: sbatch bash_scripts/extract_camsc_scratch.sbatch"
    exit 0
  fi
  echo "Partial zip found ($(numfmt --to=iec-i --suffix=B "${SZ}" 2>/dev/null || echo "${SZ} bytes")) — resuming."
fi

aria2c --check-certificate=false --file-allocation=none \
  -x 8 -s 8 -k 10M --continue=true \
  -o "${ZIP_NAME}" \
  "${CAMSC_DOWNLOAD_URL}"

SZ=$(stat -c%s "${CAMSC_ZIP}" 2>/dev/null || stat -f%z "${CAMSC_ZIP}")
echo "Zip size: ${SZ} bytes"
if [[ "${SZ}" -ge "${MIN_BYTES}" ]]; then
  echo "Download complete. Next: sbatch bash_scripts/extract_camsc_scratch.sbatch"
else
  echo "Incomplete (~$(( SZ / 1024 / 1024 / 1024 )) GiB). Re-submit this sbatch to resume."
  exit 0
fi
