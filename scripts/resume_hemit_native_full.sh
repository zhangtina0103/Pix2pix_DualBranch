#!/usr/bin/env bash
# Resume native HEMIT train (pix2pix / cut / …) in 6h chunks until epoch 80.
#
#   export MODEL=pix2pix
#   bash scripts/resume_hemit_native_full.sh
#   MODEL=cut sbatch bash_scripts/resume_hemit_native_full.sbatch
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MODE=train
export CONTINUE_TRAIN=1
export N_EPOCHS="${N_EPOCHS:-50}"
export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-30}"
_end=$((N_EPOCHS + N_EPOCHS_DECAY))

export MODEL="${MODEL:?Set MODEL=pix2pix|pix2pixhd|cut|asp|cyclegan}"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
# shellcheck source=/dev/null
source "${ROOT}/bash_scripts/_hemit_gpu.sh"
hemit_sync_gpu_env

# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_model_profiles.sh"

ckpt_dir="${ROOT}/checkpoints/${TRAIN_NAME}"
[[ -d "${ckpt_dir}" ]] || { echo "ERROR: missing ${ckpt_dir} — run train sbatch first" >&2; exit 1; }

if [[ -z "${RESUME_FROM_EPOCH:-}" ]]; then
  RESUME_FROM_EPOCH="$(
    find "${ckpt_dir}" -maxdepth 1 -name '*_net_G.pth' ! -name 'latest_net_G.pth' -printf '%f\n' 2>/dev/null \
      | sed -n 's/^\([0-9][0-9]*\)_net_G\.pth$/\1/p' | sort -n | tail -1
  )"
  [[ -n "${RESUME_FROM_EPOCH}" ]] || { echo "ERROR: no numbered checkpoint in ${ckpt_dir}" >&2; exit 1; }
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

echo "===== resume native: MODEL=${MODEL} ${TRAIN_NAME} load@${RESUME_FROM_EPOCH} train ${EPOCH_COUNT}..${_end} ====="
bash scripts/run_hemit_all.sh
