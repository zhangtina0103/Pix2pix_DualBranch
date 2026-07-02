# shellcheck shell=bash
# HEMIT paths on ORCD Engaging scratch (home quota ~200G — keep datasets/ckpts off home).
#
#   source scripts/hemit_scratch_env.sh
#
# Scratch layout:
#   ~/orcd/scratch/hemit/
#     datasets/hemit/          — pix2pix trainA/B (symlinks to HEMIT_SRC)
#     datasets/hemit_multi/    — multi-input variants he|dapi|he_dapi
#     checkpoints/             — training checkpoints (optional)
#     results/                 — test outputs + metrics (optional)
#
# Raw official HEMIT (train/input, train/label) may stay on home:
#   HEMIT_SRC=/home/zhangtin/HEMIT

if [[ -z "${HEMIT_SCRATCH_ROOT:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    HEMIT_SCRATCH_ROOT="${SCRATCH}/hemit"
  else
    HEMIT_SCRATCH_ROOT="${HOME}/orcd/scratch/hemit"
  fi
fi

export HEMIT_SCRATCH_ROOT
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT_HEMIT="${DATAROOT_HEMIT:-${HEMIT_SCRATCH_ROOT}/datasets/hemit}"
export HEMIT_MULTI_ROOT="${HEMIT_MULTI_ROOT:-${HEMIT_SCRATCH_ROOT}/datasets/hemit_multi}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${HEMIT_SCRATCH_ROOT}/checkpoints}"
export RESULTS_DIR="${RESULTS_DIR:-${HEMIT_SCRATCH_ROOT}/results}"

hemit_scratch_ensure_dirs() {
  mkdir -p \
    "${HEMIT_SCRATCH_ROOT}/datasets" \
    "${DATAROOT_HEMIT}" \
    "${HEMIT_MULTI_ROOT}" \
    "${CHECKPOINTS_DIR}" \
    "${RESULTS_DIR}"
}

hemit_multiinput_dataroot() {
  local variant="${1:-${HEMIT_MULTI_VARIANT:?variant required}}"
  echo "${HEMIT_MULTI_ROOT}/${variant}"
}

hemit_apply_scratch_env() {
  hemit_scratch_ensure_dirs
  echo "HEMIT scratch: root=${HEMIT_SCRATCH_ROOT}" >&2
  echo "  HEMIT_SRC=${HEMIT_SRC}" >&2
  echo "  DATAROOT_HEMIT=${DATAROOT_HEMIT}" >&2
  echo "  HEMIT_MULTI_ROOT=${HEMIT_MULTI_ROOT}" >&2
  echo "  CHECKPOINTS_DIR=${CHECKPOINTS_DIR}" >&2
  echo "  RESULTS_DIR=${RESULTS_DIR}" >&2
}
