#!/usr/bin/env bash
# Submit HEMIT multi-input training with logs + ckpts on scratch (not home).
#
#   bash bash_scripts/submit_hemit_multiinput_train.sh he
#   bash bash_scripts/submit_hemit_multiinput_train.sh dapi
#   bash bash_scripts/submit_hemit_multiinput_train.sh he_dapi
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VARIANT="${1:?usage: submit_hemit_multiinput_train.sh he|dapi|he_dapi}"
case "${VARIANT}" in
  he|dapi|he_dapi) ;;
  *) echo "ERROR: variant must be he, dapi, or he_dapi" >&2; exit 1 ;;
esac

# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_scratch_env.sh"
hemit_scratch_ensure_dirs
mkdir -p "${HEMIT_SCRATCH_ROOT}/logs"

DATAROOT_VARIANT="$(hemit_multiinput_dataroot "${VARIANT}")"
if [[ ! -d "${DATAROOT_VARIANT}/trainA" ]]; then
  echo "ERROR: dataroot not ready: ${DATAROOT_VARIANT}/trainA" >&2
  echo "  Prep only this variant:" >&2
  echo "    HEMIT_MULTI_VARIANTS=${VARIANT} sbatch bash_scripts/prepare_hemit_multiinput.sbatch" >&2
  exit 1
fi

n_train=$(find "${DATAROOT_VARIANT}/trainA" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "Submit train: variant=${VARIANT} tiles=${n_train}"
echo "  DATAROOT=${DATAROOT_VARIANT}"
echo "  CHECKPOINTS_DIR=${CHECKPOINTS_DIR}"
echo "  logs → ${HEMIT_SCRATCH_ROOT}/logs/"

export HEMIT_MULTI_VARIANT="${VARIANT}"
sbatch \
  --job-name="hemit_${VARIANT}" \
  --output="${HEMIT_SCRATCH_ROOT}/logs/train_${VARIANT}_%j.out" \
  --error="${HEMIT_SCRATCH_ROOT}/logs/train_${VARIANT}_%j.err" \
  bash_scripts/train_hemit_multiinput.sbatch
