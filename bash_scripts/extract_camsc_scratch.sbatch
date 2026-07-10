#!/bin/bash
# Unzip CaMSC Dropbox archive and merge with existing 20260504 batch on scratch.
#
#   sbatch bash_scripts/extract_camsc_scratch.sbatch
#
# Writes flat TIF pool → ${CAMSC_COMBINED_SRC} (default: camsc_all/)
# Then prep k-fold:
#   export CAMSC_SRC=~/orcd/scratch/camsc/camsc_all
#   sbatch bash_scripts/prepare_camsc_bf.sbatch
#SBATCH --job-name=camsc_unzip
#SBATCH --output=logs/camsc_extract_%j.out
#SBATCH --error=logs/camsc_extract_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/camsc_scratch_env.sh"
cd "${CAMSC_SCRATCH_ROOT}"

[[ -f "${CAMSC_ZIP}" ]] || {
  echo "ERROR: missing ${CAMSC_ZIP}; run download_camsc_scratch.sbatch first" >&2
  exit 1
}

echo "Extracting ${CAMSC_ZIP} → ${CAMSC_NEW_SRC} ..."
mkdir -p "${CAMSC_NEW_SRC}"
if [[ -z "$(find "${CAMSC_NEW_SRC}" -maxdepth 2 -name '*.tif' -print -quit 2>/dev/null)" ]]; then
  unzip -q -o "${CAMSC_ZIP}" -d "${CAMSC_NEW_SRC}"
else
  echo "New batch already extracted under ${CAMSC_NEW_SRC}"
fi

echo "Flattening new TIFs into ${CAMSC_COMBINED_SRC} ..."
mkdir -p "${CAMSC_COMBINED_SRC}"

if [[ -d "${CAMSC_OLD_SRC}" ]]; then
  echo "Copying old batch: ${CAMSC_OLD_SRC}"
  cp -n "${CAMSC_OLD_SRC}"/*.tif "${CAMSC_COMBINED_SRC}/" 2>/dev/null || true
else
  echo "WARN: old batch not found at ${CAMSC_OLD_SRC} (skipping)"
fi

NEW_COUNT=0
while IFS= read -r -d '' f; do
  base="$(basename "${f}")"
  if [[ -e "${CAMSC_COMBINED_SRC}/${base}" ]]; then
    echo "  [skip collision] ${base}"
  else
    cp "${f}" "${CAMSC_COMBINED_SRC}/${base}"
    NEW_COUNT=$((NEW_COUNT + 1))
  fi
done < <(find "${CAMSC_NEW_SRC}" -name '*.tif' -print0)

TOTAL=$(find "${CAMSC_COMBINED_SRC}" -maxdepth 1 -name '*.tif' | wc -l | tr -d ' ')
echo ""
echo "Added ${NEW_COUNT} new TIFs (skipped filename collisions)."
echo "Combined pool: ${TOTAL} TIFs in ${CAMSC_COMBINED_SRC}"
echo ""
echo "Next:"
echo "  export CAMSC_SRC=${CAMSC_COMBINED_SRC}"
echo "  sbatch bash_scripts/prepare_camsc_bf.sbatch"
