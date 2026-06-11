#!/usr/bin/env bash
# Orion-Lite — same entrypoint pattern as run_hemit_all.sh.
#
#   MODEL=pix2pix MODE=train bash scripts/run_orion_all.sh
#   MODEL=vanilla_fm MODE=train|test|metrics bash scripts/run_orion_all.sh
#   MODE=prepare bash scripts/run_orion_all.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export REPO_ROOT="${ROOT}"

export MODEL="${MODEL:-pix2pix}"
export MODE="${MODE:-all}"
export ORION_SRC="${ORION_SRC:-./data/orion/ORIONCRC_dataset_tile_20x}"
export DATAROOT="${DATAROOT:-./datasets/orion_lite}"

die() { echo "ERROR: $*" >&2; exit 1; }

is_native() {
  case "${MODEL}" in
    dualbranch|pix2pix|pix2pixhd|resnet9|resnet6|unet256|unet128|unet1024|swint|swint_unet|cut|asp|cyclegan|vanilla_fm) return 0 ;;
    *) return 1 ;;
  esac
}

echo "===== Orion-Lite  MODEL=${MODEL}  MODE=${MODE}  DATAROOT=${DATAROOT} ====="

if is_native; then
  source "${ROOT}/scripts/orion_model_profiles.sh"
  if [[ "${MODEL}" == "vanilla_fm" ]]; then
    source "${ROOT}/scripts/vanilla_fm_env.sh"
    if [[ "${VANILLA_FM_ENV_LOCKED:-0}" != "1" ]]; then
      vanilla_fm_apply_orion_joint_perc_scratch_env
    fi
    vanilla_fm_verify_locked_env
    vanilla_fm_print_train_env
  else
    echo "Profile: PY_MODEL=${PY_MODEL} TRAIN_NAME=${TRAIN_NAME} NUM_TEST=${NUM_TEST}"
  fi
  [[ -f "${ROOT}/bash_scripts/_hemit_gpu.sh" ]] && source "${ROOT}/bash_scripts/_hemit_gpu.sh" && hemit_sync_gpu_env
  export GPU_IDS="${GPU_IDS:-0}"
  bash "${ROOT}/scripts/run_orion_native.sh"
else
  die "Unknown MODEL=${MODEL}. Use: pix2pix | cut | asp | cyclegan | vanilla_fm"
fi

echo "Done."
