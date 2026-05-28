#!/usr/bin/env bash
# Reproduce the official HEMIT Dual-Branch README pipeline (no custom settings).
#
# From Pix2pix_DualBranch root, conda env activated:
#
#   # A) Fastest: their pretrained weights (download from README Google Drive)
#   #    Unzip so checkpoints/hemit_SwinTResnet_New/ contains *.pth files
#   MODE=eval_pretrained bash scripts/run_hemit_reproduce.sh
#
#   # B) Train from scratch (README command), then test + post_process
#   MODE=all bash scripts/run_hemit_reproduce.sh
#
#   # C) Steps separately
#   MODE=prepare|train|test|metrics bash scripts/run_hemit_reproduce.sh
#
# Data: ~/virtual-staining/data/hemit/{train,val,test}/{input,label}/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-all}"
HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
DATAROOT="${DATAROOT:-./datasets/hemit}"
GPU_IDS="${GPU_IDS:-0}"

# README experiment names
TRAIN_NAME="${TRAIN_NAME:-hemit_SwinTResnet_New_2}"
PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_SwinTResnet_New}"

# Epochs
N_EPOCHS="${N_EPOCHS:-50}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-30}"
PRETRAINED_EPOCH="${PRETRAINED_EPOCH:-20}"
TEST_EPOCH="${TEST_EPOCH:-$((N_EPOCHS + N_EPOCHS_DECAY))}"   # 80 by default

die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

need python

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

prepare() {
  echo "==> prepare (symlink) dataset at ${DATAROOT}"
  [[ -d "${HEMIT_SRC}/train" ]] || die "HEMIT_SRC not found: ${HEMIT_SRC} (set HEMIT_SRC=...)"
  # Symlinks by default; pass PREPARE_COPY=1 to copy instead.
  local copy_flag=()
  [[ "${PREPARE_COPY:-0}" == "1" ]] && copy_flag=(--copy)
  python scripts/prepare_hemit_data.py \
    --src "${HEMIT_SRC}" \
    --dst "${DATAROOT}" \
    "${copy_flag[@]}"
}

train() {
  echo "==> train (README flags) name=${TRAIN_NAME}"
  python train.py \
    --dataroot "${DATAROOT}" \
    --name "${TRAIN_NAME}" \
    --model pix2pix \
    --direction AtoB \
    --display_id 0 \
    --lr 0.00003 \
    --lambda_L1 30 \
    --no_flip \
    --netG SwinTResnet \
    --n_epochs "${N_EPOCHS}" \
    --n_epochs_decay "${N_EPOCHS_DECAY}" \
    --lr_policy step \
    --batch_size 2 \
    --loss_type L1 \
    --val_freq 5
}

test_one() {
  local name="$1"
  local epoch="$2"
  local num_test="$3"
  echo "==> test name=${name} epoch=${epoch} num_test=${num_test}"
  python test.py \
    --dataroot "${DATAROOT}" \
    --name "${name}" \
    --model pix2pix \
    --direction AtoB \
    --epoch "${epoch}" \
    --num_test "${num_test}" \
    --eval \
    --netG SwinTResnet
    # test.py sets display_id=-1 internally (no visdom)
}

metrics_one() {
  local name="$1"
  local epoch="$2"
  local srcdir="results/${name}/test_${epoch}/"
  echo "==> metrics srcdir=${srcdir}"
  [[ -d "${srcdir}" ]] || die "missing results dir: ${srcdir} (run MODE=test first)"
  python post_process.py --srcdir "${srcdir}"
}

eval_pretrained() {
  echo "==> eval_pretrained (README checkpoint) name=${PRETRAINED_NAME} epoch=${PRETRAINED_EPOCH}"
  [[ -d "checkpoints/${PRETRAINED_NAME}" ]] || die "missing checkpoints/${PRETRAINED_NAME} (download and unzip README checkpoint)"
  test_one "${PRETRAINED_NAME}" "${PRETRAINED_EPOCH}" 945
  metrics_one "${PRETRAINED_NAME}" "${PRETRAINED_EPOCH}"
}

test_trained() {
  test_one "${TRAIN_NAME}" "${TEST_EPOCH}" 945
}

metrics_trained() {
  metrics_one "${TRAIN_NAME}" "${TEST_EPOCH}"
}

run_mode() {
  case "$1" in
    prepare) prepare ;;
    train) train ;;
    test) test_trained ;;
    metrics) metrics_trained ;;
    eval_pretrained) eval_pretrained ;;
    all)
      prepare
      train
      test_trained
      metrics_trained
      ;;
    *) die "unknown MODE='$1' (use prepare|train|test|metrics|eval_pretrained|all)" ;;
  esac
}

IFS='|' read -r -a _modes <<< "${MODE}"
for m in "${_modes[@]}"; do
  run_mode "${m}"
done

echo "Done."

