#!/bin/bash
# Unzip ORION zip on scratch (after download_orion_scratch.sbatch).
#
#   sbatch bash_scripts/extract_orion_scratch.sbatch
#SBATCH --job-name=orion_unzip
#SBATCH --output=logs/orion_extract_%j.out
#SBATCH --error=logs/orion_extract_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/orion_scratch_env.sh"
cd "${ORION_SCRATCH_ROOT}"

[[ -f "${ORION_ZIP}" ]] || { echo "ERROR: missing ${ORION_ZIP}; run download_orion_scratch.sbatch first" >&2; exit 1; }

if [[ -f "${ORION_SRC}/train_dataframe.csv" ]]; then
  echo "Already extracted at ${ORION_SRC}"
else
  echo "Extracting ${ORION_ZIP} ..."
  unzip -q "${ORION_ZIP}"
fi

find "${ORION_SCRATCH_ROOT}" -maxdepth 3 -name train_dataframe.csv | head -5
echo "Next: sbatch bash_scripts/prepare_orion_lite.sbatch"
