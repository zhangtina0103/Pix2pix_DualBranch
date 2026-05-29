#!/usr/bin/env bash
# Resume Vanilla FM after a crash before the epoch-80 checkpoint save.
# Loads 75_net_G.pth (default), runs epochs 76–80, writes 80_net_G.pth.
#
# Local / interactive GPU:
#   bash scripts/resume_hemit_vanilla_fm.sh
#
# Cluster:
#   sbatch bash_scripts/resume_hemit_vanilla_fm.sbatch
#
# Overrides:
#   RESUME_FROM_EPOCH=70 EPOCH_COUNT=71 TRAIN_LR=0.00002 bash scripts/resume_hemit_vanilla_fm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MODEL=vanilla_fm
export MODE=train
export CONTINUE_TRAIN=1

export N_EPOCHS="${N_EPOCHS:-50}"
export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-30}"
_end=$((N_EPOCHS + N_EPOCHS_DECAY))

export RESUME_FROM_EPOCH="${RESUME_FROM_EPOCH:-75}"
export EPOCH_COUNT="${EPOCH_COUNT:-$((RESUME_FROM_EPOCH + 1))}"

if (( EPOCH_COUNT > _end )); then
  echo "ERROR: EPOCH_COUNT=${EPOCH_COUNT} > end epoch ${_end} (nothing to train)" >&2
  exit 1
fi

# Step LR already decayed once by epoch 50; fresh resume must not restart at 2e-4.
export TRAIN_LR="${TRAIN_LR:-0.00002}"
export DISPLAY_ID="${DISPLAY_ID:--1}"
export FM_CHANNELS="${FM_CHANNELS:-96,192,272}"
export FM_RESBLOCKS="${FM_RESBLOCKS:-2}"
export FM_STEPS="${FM_STEPS:-25}"
export TRAIN_NAME="${TRAIN_NAME:-hemit_vanilla_fm_joint}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export GPU_IDS="${GPU_IDS:-0}"

echo "===== resume vanilla_fm: ${RESUME_FROM_EPOCH} -> ${_end} (epochs ${EPOCH_COUNT}..${_end}) ====="
echo "  TRAIN_NAME=${TRAIN_NAME}  TRAIN_LR=${TRAIN_LR}  DISPLAY_ID=${DISPLAY_ID}"

exec bash "${ROOT}/scripts/run_hemit_vanilla_fm.sh"
