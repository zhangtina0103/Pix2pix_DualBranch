# Orion-Lite profiles — same train.py → test.py → post_process.py as HEMIT.
# Source hemit defaults, then rename checkpoints/results to orion_lite_*.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ORION_DATAROOT="${ORION_DATAROOT:-${DATAROOT:-./datasets/orion_lite}}"
export DATAROOT="${ORION_DATAROOT}"

# Preserve FM profile TRAIN_NAME when locked (apply_orion_* set it before this runs).
_locked_train_name="${TRAIN_NAME:-}"
_locked_pretrained_name="${PRETRAINED_NAME:-}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"

_orion_rename() {
  local v="$1"
  v="${v/hemit_/orion_lite_}"
  v="${v/hemit-/orion_lite-}"
  printf '%s' "$v"
}

if [[ "${MODEL}" == "vanilla_fm" && "${VANILLA_FM_ENV_LOCKED:-0}" == "1" && -n "${_locked_train_name}" ]]; then
  TRAIN_NAME="${_locked_train_name}"
  PRETRAINED_NAME="${_locked_pretrained_name:-${_locked_train_name}}"
else
  TRAIN_NAME="$(_orion_rename "${TRAIN_NAME}")"
  PRETRAINED_NAME="$(_orion_rename "${PRETRAINED_NAME}")"
fi

# hemit_model_profiles appends _512 before we restore locked FM names — re-apply here.
if [[ "${HEMIT_TRAIN_SIZE:-512}" != "1024" ]]; then
  if [[ "${TRAIN_NAME}" != *"_${HEMIT_TRAIN_SIZE}" ]]; then
    TRAIN_NAME="${TRAIN_NAME}_${HEMIT_TRAIN_SIZE}"
    PRETRAINED_NAME="${PRETRAINED_NAME}_${HEMIT_TRAIN_SIZE}"
  fi
fi

if [[ -f "${DATAROOT}/meta.json" ]]; then
  NUM_TEST="$(python3 -c "import json; print(json.load(open('${DATAROOT}/meta.json'))['test_count'])")"
fi
export TRAIN_NAME PRETRAINED_NAME NUM_TEST DATAROOT

if [[ "${MODEL}" == "vanilla_fm" && "${VANILLA_FM_ENV_LOCKED:-0}" == "1" ]]; then
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
fi
