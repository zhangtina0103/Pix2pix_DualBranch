#!/usr/bin/env bash
# Reproduce the official HEMIT Dual-Branch README pipeline (no custom settings).
#
# From Pix2pix_DualBranch root, venv activated:
#   source .venv-hemit/bin/activate && source bash_scripts/_cuda.sh
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
HEMIT_SRC="${HEMIT_SRC:-}"
DATAROOT="${DATAROOT:-./datasets/hemit}"
GPU_IDS="${GPU_IDS:-0}"

# README experiment names
TRAIN_NAME="${TRAIN_NAME:-hemit_SwinTResnet_New_2}"
PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_SwinTResnet_New}"
TEST_EPOCH="${TEST_EPOCH:-20}"          # README eval_pretrained example
NUM_TEST="${NUM_TEST:-945}"             # README test example

if [[ -z "$HEMIT_SRC" ]]; then
  for candidate in \
    ../virtual-staining/data/hemit \
    "$HOME/virtual-staining/data/hemit" \
    ../vs_v2/data/hemit; do
    if [[ -d "$candidate/train/input" ]]; then
      HEMIT_SRC="$candidate"
      break
    fi
  done
fi

prepare_data() {
  if [[ -z "$HEMIT_SRC" || ! -d "$HEMIT_SRC/train/input" ]]; then
    echo "ERROR: HEMIT_SRC must point to hemit/ with train/input and train/label"
    exit 1
  fi
  if [[ ! -d "$DATAROOT/trainA" ]] || [[ -z "$(ls -A "$DATAROOT/trainA" 2>/dev/null || true)" ]]; then
    echo ">>> prepare: $HEMIT_SRC → $DATAROOT (trainA/B, valA/B, testA/B)"
    python scripts/prepare_hemit_data.py --src "$HEMIT_SRC" --dst "$DATAROOT"
  else
    echo ">>> prepare: skipped ($DATAROOT/trainA exists)"
  fi
}

# Exactly as README "Example Training"
run_train() {
  echo ">>> train (README): name=$TRAIN_NAME"
  python train.py \
    --dataroot "$DATAROOT" \
    --name "$TRAIN_NAME" \
    --model pix2pix \
    --direction AtoB \
    --display_id 0 \
    --lr 0.00003 \
    --lambda_L1 30 \
    --no_flip \
    --netG SwinTResnet \
    --n_epochs 50 \
    --n_epochs_decay 30 \
    --lr_policy step \
    --batch_size 2 \
    --loss_type L1 \
    --val_freq 5 \
    --gpu_ids "$GPU_IDS"
}

# Exactly as README "Example testing" (override --name / --epoch via env)
run_test() {
  local name="${1:?}"
  local epoch="${2:?}"
  echo ">>> test (README): name=$name epoch=$epoch num_test=$NUM_TEST"
  python test.py \
    --dataroot "$DATAROOT" \
    --name "$name" \
    --model pix2pix \
    --direction AtoB \
    --epoch "$epoch" \
    --num_test "$NUM_TEST" \
    --eval \
    --netG SwinTResnet \
    --gpu_ids "$GPU_IDS"
}

# Exactly as README post_process
run_metrics() {
  local srcdir="${1:?}"
  echo ">>> post_process: $srcdir"
  python post_process.py --srcdir "$srcdir"
  if [[ -f "$srcdir/score.csv" ]]; then
    echo "--- score.csv (last 3 rows) ---"
    tail -3 "$srcdir/score.csv"
  fi
}

echo "=============================================="
echo " HEMIT reproduce (official README only)"
echo " MODE=$MODE  DATAROOT=$DATAROOT"
echo " HEMIT_SRC=${HEMIT_SRC:-<not set>}"
echo "=============================================="

case "$MODE" in
  prepare)
    prepare_data
    ;;
  train)
    prepare_data
    run_train
    echo "Checkpoints: checkpoints/${TRAIN_NAME}/"
    echo "Val SSIM log: checkpoints/${TRAIN_NAME}/validation_train.csv"
    ;;
  test)
    prepare_data
    name="${NAME:-$TRAIN_NAME}"
    epoch="${TEST_EPOCH:-80}"
    run_test "$name" "$epoch"
    ;;
  metrics)
    srcdir="${RESULTS_DIR:-results/${PRETRAINED_NAME}/test_${TEST_EPOCH}}"
    run_metrics "$srcdir"
    ;;
  eval_pretrained)
    prepare_data
    if [[ ! -d "checkpoints/${PRETRAINED_NAME}" ]]; then
      echo "ERROR: Download README checkpoint into checkpoints/${PRETRAINED_NAME}/"
      echo "  https://drive.google.com/file/d/1HNc-dj2ATN7gdAyOCy-lWe8_YQse2CTd/view"
      exit 1
    fi
    run_test "$PRETRAINED_NAME" "$TEST_EPOCH"
    run_metrics "results/${PRETRAINED_NAME}/test_${TEST_EPOCH}"
    ;;
  all)
    prepare_data
    run_train
    # After 50+30 epochs, weights saved at epoch 80 (and every 5 epochs)
    TRAIN_TEST_EPOCH="${TRAIN_TEST_EPOCH:-80}"
    run_test "$TRAIN_NAME" "$TRAIN_TEST_EPOCH"
    run_metrics "results/${TRAIN_NAME}/test_${TRAIN_TEST_EPOCH}"
    ;;
  *)
    echo "Unknown MODE=$MODE (prepare|train|test|metrics|eval_pretrained|all)"
    exit 1
    ;;
esac

echo "Done: $(date)"
