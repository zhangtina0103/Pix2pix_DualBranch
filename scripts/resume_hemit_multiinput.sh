#!/usr/bin/env bash
# Resume HEMIT multi-input FM training (6h walltime chunks until epoch 80).
#
#   export HEMIT_MULTI_VARIANT=he
#   export RESUME_FROM_EPOCH=45      # optional; default = latest saved *_net_G.pth
#   bash scripts/resume_hemit_multiinput.sh
#
# Cluster:
#   HEMIT_MULTI_VARIANT=he sbatch bash_scripts/resume_hemit_multiinput.sbatch
#   HEMIT_MULTI_VARIANT=he_dapi RESUME_FROM_EPOCH=50 sbatch bash_scripts/resume_hemit_multiinput.sbatch
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MODEL=vanilla_fm
export MODE=train
export CONTINUE_TRAIN=1
export HEMIT_MULTI_VARIANT="${HEMIT_MULTI_VARIANT:-he}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_multiinput_env.sh"
hemit_multiinput_apply_env
export VANILLA_FM_ENV_LOCKED=1

# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_model_profiles.sh"
export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
vanilla_fm_verify_locked_env

ckpt_dir="${CHECKPOINTS_DIR}/${TRAIN_NAME}"
if [[ ! -d "${ckpt_dir}" ]]; then
  echo "ERROR: missing ${ckpt_dir}" >&2
  echo "  Train first: HEMIT_MULTI_VARIANT=${HEMIT_MULTI_VARIANT} sbatch bash_scripts/train_hemit_multiinput.sbatch" >&2
  exit 1
fi

if [[ -z "${RESUME_FROM_EPOCH:-}" ]]; then
  RESUME_FROM_EPOCH="$(
    find "${ckpt_dir}" -maxdepth 1 -name '*_net_G.pth' ! -name 'latest_net_G.pth' -printf '%f\n' 2>/dev/null \
      | sed -n 's/^\([0-9][0-9]*\)_net_G\.pth$/\1/p' | sort -n | tail -1
  )"
  if [[ -z "${RESUME_FROM_EPOCH}" ]] && [[ -f "${ckpt_dir}/latest_net_G.pth" ]]; then
    echo "WARN: only latest_net_G.pth found; set RESUME_FROM_EPOCH explicitly" >&2
    exit 1
  fi
  if [[ -z "${RESUME_FROM_EPOCH}" ]]; then
    echo "ERROR: no checkpoint in ${ckpt_dir}" >&2
    exit 1
  fi
fi

export RESUME_FROM_EPOCH
export CONTINUE_TRAIN=1
export EPOCH_COUNT="${EPOCH_COUNT:-$((RESUME_FROM_EPOCH + 1))}"
_end=$((N_EPOCHS + N_EPOCHS_DECAY))

if (( EPOCH_COUNT > _end )); then
  echo "Training complete: ${TRAIN_NAME} @ epoch ${_end}"
  exit 0
fi

if (( RESUME_FROM_EPOCH >= N_EPOCHS )); then
  export TRAIN_LR="${TRAIN_LR:-0.00002}"
else
  export TRAIN_LR="${TRAIN_LR:-0.0002}"
fi

vanilla_fm_print_train_env
echo "===== resume multi-input: variant=${HEMIT_MULTI_VARIANT} ${TRAIN_NAME} load@${RESUME_FROM_EPOCH} train epochs ${EPOCH_COUNT}..${_end} ====="
echo "  DATAROOT=${DATAROOT}"
echo "  CHECKPOINTS_DIR=${CHECKPOINTS_DIR}"

exec bash "${ROOT}/scripts/run_hemit_vanilla_fm.sh"
