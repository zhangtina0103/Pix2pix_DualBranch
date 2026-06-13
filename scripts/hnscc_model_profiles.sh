# HNSCC profiles — same train.py → test.py flow as Orion/HEMIT.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HNSCC_DATAROOT="${HNSCC_DATAROOT:-${DATAROOT:-./datasets/hnscc}}"
export DATAROOT="${HNSCC_DATAROOT}"

_locked_train_name="${TRAIN_NAME:-}"
_locked_pretrained_name="${PRETRAINED_NAME:-}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"

_hnscc_rename() {
  local v="$1"
  v="${v/hemit_/hnscc_}"
  v="${v/hemit-/hnscc-}"
  printf '%s' "$v"
}

if [[ "${MODEL}" == "vanilla_fm" && "${VANILLA_FM_ENV_LOCKED:-0}" == "1" && -n "${_locked_train_name}" ]]; then
  TRAIN_NAME="${_locked_train_name}"
  PRETRAINED_NAME="${_locked_pretrained_name:-${_locked_train_name}}"
else
  TRAIN_NAME="$(_hnscc_rename "${TRAIN_NAME}")"
  PRETRAINED_NAME="$(_hnscc_rename "${PRETRAINED_NAME}")"
fi

if [[ "${HEMIT_TRAIN_SIZE:-512}" != "1024" ]]; then
  if [[ "${TRAIN_NAME}" != *"_${HEMIT_TRAIN_SIZE}" ]]; then
    TRAIN_NAME="${TRAIN_NAME}_${HEMIT_TRAIN_SIZE}"
    PRETRAINED_NAME="${PRETRAINED_NAME}_${HEMIT_TRAIN_SIZE}"
  fi
fi

if [[ -f "${DATAROOT}/meta.json" ]]; then
  NUM_TEST="$(python3 -c "import json; print(json.load(open('${DATAROOT}/meta.json'))['test_count'])")"
  HNSCC_OUTPUT_NC="$(python3 -c "import json; print(json.load(open('${DATAROOT}/meta.json')).get('output_nc', 4))")"
fi

# HNSCC: 3ch H&E → 4ch mIF; CycleGAN identity loss needs input_nc == output_nc.
if [[ "${MODEL}" == "cyclegan" && "${HNSCC_OUTPUT_NC:-4}" != "3" ]]; then
  export LAMBDA_IDENTITY=0
fi

export TRAIN_NAME PRETRAINED_NAME NUM_TEST DATAROOT HNSCC_OUTPUT_NC

if [[ "${MODEL}" == "vanilla_fm" && "${VANILLA_FM_ENV_LOCKED:-0}" == "1" ]]; then
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
fi
