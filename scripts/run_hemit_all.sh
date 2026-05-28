#!/usr/bin/env bash
# HEMIT — one entrypoint, two backends:
#
#   A) Native HEMIT (joint 3ch): train.py → test.py → post_process.py
#      MODEL=dualbranch | pix2pix | cut | asp | cyclegan | resnet9 | ...
#
#   B) hemit/ flow matching only: MODEL=fm | fm_plus
#
# Examples:
#   MODEL=dualbranch MODE=all bash scripts/run_hemit_all.sh
#   MODEL=cut MODE=train|test|metrics bash scripts/run_hemit_all.sh
#   MODEL=fm_plus MODE=train bash scripts/run_hemit_all.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MODEL="${MODEL:-dualbranch}"
export MODE="${MODE:-all}"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export VS_DATA_ROOT="${VS_DATA_ROOT:-${HEMIT_SRC}}"
export GPU_IDS="${GPU_IDS:-0}"
export REPO_ROOT="${ROOT}"

die() { echo "ERROR: $*" >&2; exit 1; }

is_native() {
  case "${MODEL}" in
    dualbranch|pix2pix|resnet9|resnet6|unet256|unet128|unet1024|swint|swint_unet|cut|asp|cyclegan|vanilla_fm) return 0 ;;
    *) return 1 ;;
  esac
}

is_hemit_fm() {
  case "${MODEL}" in
    fm|fm_plus|vanilla_fm) return 0 ;;
    *) return 1 ;;
  esac
}

run_hemit_comparison() {
  export MODEL
  case "$1" in
    prepare)
      echo "==> ${MODEL}: uses HEMIT_SRC=${VS_DATA_ROOT} (input/label); no trainA/B prep"
      ;;
    train)
      bash "${ROOT}/scripts/hemit_comparison/train.sh"
      ;;
    metrics|eval)
      bash "${ROOT}/scripts/hemit_comparison/eval.sh"
      ;;
    all)
      die "MODE=all not supported for ${MODEL}; use MODE=train then MODE=metrics"
      ;;
    test)
      echo "==> ${MODEL}: inference is part of hemit eval (MODE=metrics)"
      ;;
    *) die "unknown MODE='$1'" ;;
  esac
}

echo "===== HEMIT  MODEL=${MODEL}  MODE=${MODE} ====="

if [[ "${MODEL}" == "comparison" ]]; then
  export MODEL=comparison
  run_hemit_comparison metrics
elif is_native; then
  bash "${ROOT}/scripts/run_hemit_native.sh"
elif is_hemit_fm; then
  IFS='|' read -r -a _modes <<< "${MODE}"
  for m in "${_modes[@]}"; do run_hemit_comparison "${m}"; done
else
  die "Unknown MODEL=${MODEL}. Native: dualbranch|pix2pix|cut|asp|cyclegan|...  FM: fm|fm_plus"
fi

echo "Done."
