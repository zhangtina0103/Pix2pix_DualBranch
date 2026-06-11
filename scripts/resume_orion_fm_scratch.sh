#!/usr/bin/env bash
# Resume Orion-Lite FM in 6h chunks until epoch 80.
#
#   export RESUME_PROFILE=vanilla_fm   # or cross_attn
#   bash scripts/resume_orion_fm_scratch.sh
#
# Cluster:
#   sbatch bash_scripts/resume_orion_lite_vanilla_fm_scratch.sbatch
#   sbatch bash_scripts/resume_orion_lite_fm_cross_attn_scratch.sbatch
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MODEL=vanilla_fm
export MODE=train
export CONTINUE_TRAIN=1
export DISPLAY_ID="${DISPLAY_ID:--1}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/orion_scratch_env.sh"
# shellcheck source=/dev/null
source "${ROOT}/scripts/vanilla_fm_env.sh"

export RESUME_PROFILE="${RESUME_PROFILE:?Set RESUME_PROFILE=vanilla_fm|cross_attn}"

case "${RESUME_PROFILE}" in
  vanilla_fm|joint_perc|joint_perc_scratch)
    vanilla_fm_apply_orion_joint_perc_scratch_env
    ;;
  cross_attn|cross_attn_scratch)
    vanilla_fm_apply_orion_fm_cross_attn_scratch_env
    unset FM_CROSS_ATTN_DECODER
    export FM_CROSS_ATTN_HEADS="${FM_CROSS_ATTN_HEADS:-4}"
    ;;
  *)
    echo "ERROR: unknown RESUME_PROFILE=${RESUME_PROFILE} (use vanilla_fm or cross_attn)" >&2
    exit 1
    ;;
esac

export VANILLA_FM_ENV_LOCKED=1
# shellcheck source=/dev/null
source "${ROOT}/scripts/orion_model_profiles.sh"
export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
vanilla_fm_verify_locked_env

ckpt_dir="${ROOT}/checkpoints/${TRAIN_NAME}"
[[ -d "${ckpt_dir}" ]] || { echo "ERROR: missing ${ckpt_dir}" >&2; exit 1; }

if [[ -z "${RESUME_FROM_EPOCH:-}" ]]; then
  RESUME_FROM_EPOCH="$(
    find "${ckpt_dir}" -maxdepth 1 -name '*_net_G.pth' ! -name 'latest_net_G.pth' -printf '%f\n' 2>/dev/null \
      | sed -n 's/^\([0-9][0-9]*\)_net_G\.pth$/\1/p' | sort -n | tail -1
  )"
  if [[ -z "${RESUME_FROM_EPOCH}" ]] && [[ -f "${ckpt_dir}/latest_net_G.pth" ]]; then
    echo "WARN: only latest_net_G.pth — set RESUME_FROM_EPOCH to last save_epoch_freq (default 5)" >&2
    exit 1
  fi
  [[ -n "${RESUME_FROM_EPOCH}" ]] || { echo "ERROR: no numbered ckpt in ${ckpt_dir}" >&2; exit 1; }
fi

export RESUME_FROM_EPOCH
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
echo "===== resume orion FM: ${TRAIN_NAME} load@${RESUME_FROM_EPOCH} train ${EPOCH_COUNT}..${_end} ====="

exec bash "${ROOT}/scripts/run_orion_native.sh"
