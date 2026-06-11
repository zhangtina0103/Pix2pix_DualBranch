#!/usr/bin/env bash
# Resume Orion native train (pix2pix / cut / asp / cyclegan) in 6h chunks until epoch 80.
#
#   export MODEL=cyclegan
#   bash scripts/resume_orion_native.sh
#   MODEL=cyclegan sbatch bash_scripts/resume_orion_lite_gan.sbatch
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MODE=train
export CONTINUE_TRAIN=1
export N_EPOCHS="${N_EPOCHS:-50}"
export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-30}"
_end=$((N_EPOCHS + N_EPOCHS_DECAY))

export MODEL="${MODEL:?Set MODEL=pix2pix|cut|asp|cyclegan}"
# shellcheck source=/dev/null
source "${ROOT}/scripts/orion_scratch_env.sh"
# shellcheck source=/dev/null
source "${ROOT}/scripts/orion_model_profiles.sh"

ckpt_dir="${ROOT}/checkpoints/${TRAIN_NAME}"
[[ -d "${ckpt_dir}" ]] || { echo "ERROR: missing ${ckpt_dir} — run train sbatch first" >&2; exit 1; }

if [[ -z "${RESUME_FROM_EPOCH:-}" ]]; then
  RESUME_FROM_EPOCH="$(
    find "${ckpt_dir}" -maxdepth 1 -name '[0-9]*_net_*.pth' ! -name 'latest_*' -printf '%f\n' 2>/dev/null \
      | sed -n 's/^\([0-9][0-9]*\)_net_.*\.pth$/\1/p' | sort -n | tail -1
  )"
  if [[ -z "${RESUME_FROM_EPOCH}" && -f "${ckpt_dir}/latest_net_G_A.pth" ]]; then
    echo "WARN: only latest_net_G_A.pth — set RESUME_FROM_EPOCH to last save_epoch_freq (default 5)" >&2
  fi
  [[ -n "${RESUME_FROM_EPOCH}" ]] || {
    echo "ERROR: no numbered checkpoint in ${ckpt_dir}" >&2
    ls -la "${ckpt_dir}" >&2
    exit 1
  }
fi

export RESUME_FROM_EPOCH
export EPOCH_COUNT="${EPOCH_COUNT:-$((RESUME_FROM_EPOCH + 1))}"
if (( EPOCH_COUNT > _end )); then
  echo "Training complete: ${TRAIN_NAME} @ epoch ${_end}"
  exit 0
fi

if (( RESUME_FROM_EPOCH >= N_EPOCHS )); then
  export TRAIN_LR="${TRAIN_LR:-0.00002}"
fi

echo "===== resume orion: MODEL=${MODEL} ${TRAIN_NAME} load@${RESUME_FROM_EPOCH} train ${EPOCH_COUNT}..${_end} ====="
bash scripts/run_orion_all.sh
