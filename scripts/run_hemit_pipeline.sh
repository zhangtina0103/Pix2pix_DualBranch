#!/usr/bin/env bash
# End-to-end HEMIT Dual-Branch pipeline: prepare data → train → test → metrics (score.csv)
#
# Run from Pix2pix_DualBranch repo root (with your conda env activated):
#   bash scripts/run_hemit_pipeline.sh
#
# Quick smoke test (~few minutes on GPU):
#   QUICK=1 bash scripts/run_hemit_pipeline.sh
#
# Env overrides:
#   HEMIT_SRC=../vs_v2/data/hemit     # source HEMIT (input/label layout)
#   DATAROOT=./datasets/hemit         # pix2pix A/B layout
#   NAME=hemit_run                    # experiment name
#   GPU_IDS=0                         # or 0,1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HEMIT_SRC="${HEMIT_SRC:-}"
if [[ -z "$HEMIT_SRC" ]]; then
  for candidate in ../vs_v2/data/hemit ../data/hemit ./data/hemit; do
    if [[ -d "$candidate/train/input" ]]; then
      HEMIT_SRC="$candidate"
      break
    fi
  done
fi
if [[ -z "$HEMIT_SRC" || ! -d "$HEMIT_SRC/train/input" ]]; then
  echo "ERROR: Set HEMIT_SRC to HEMIT root (train/input, train/label, ...)."
  echo "  Example: HEMIT_SRC=/path/to/data/hemit bash scripts/run_hemit_pipeline.sh"
  exit 1
fi

DATAROOT="${DATAROOT:-./datasets/hemit}"
NAME="${NAME:-hemit_dualbranch}"
GPU_IDS="${GPU_IDS:-0}"
QUICK="${QUICK:-0}"

# --- defaults: paper README vs quick smoke ---
if [[ "$QUICK" == "1" ]]; then
  NET_G="${NET_G:-resnet_9blocks}"
  N_EPOCHS="${N_EPOCHS:-2}"
  N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-0}"
  SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-1}"
  VAL_FREQ="${VAL_FREQ:-1}"
  BATCH_SIZE="${BATCH_SIZE:-1}"
  NUM_TEST="${NUM_TEST:-5}"
  echo ">>> QUICK=1 smoke mode (few epochs, resnet_9blocks, 5 test images)"
else
  NET_G="${NET_G:-SwinTResnet}"
  N_EPOCHS="${N_EPOCHS:-50}"
  N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-30}"
  SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
  VAL_FREQ="${VAL_FREQ:-5}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  NUM_TEST="${NUM_TEST:-99999}"
fi

LR="${LR:-0.00003}"
LAMBDA_L1="${LAMBDA_L1:-30}"
LOAD_SIZE="${LOAD_SIZE:-1024}"
CROP_SIZE="${CROP_SIZE:-1024}"

echo "=============================================="
echo " HEMIT Dual-Branch pipeline"
echo " Repo     : $ROOT"
echo " HEMIT src: $HEMIT_SRC"
echo " Dataroot : $DATAROOT"
echo " Name     : $NAME"
echo " GPU      : $GPU_IDS"
echo " netG     : $NET_G  epochs=${N_EPOCHS}+${N_EPOCHS_DECAY}  batch=$BATCH_SIZE"
echo " Started  : $(date)"
echo "=============================================="

# --- 1) Prepare pix2pix folders (trainA/B, valA/B, testA/B) ---
if [[ ! -d "$DATAROOT/trainA" ]] || [[ -z "$(ls -A "$DATAROOT/trainA" 2>/dev/null || true)" ]]; then
  echo ""
  echo ">>> [1/4] Preparing dataroot (symlinks)..."
  python scripts/prepare_hemit_data.py --src "$HEMIT_SRC" --dst "$DATAROOT"
else
  echo ""
  echo ">>> [1/4] Dataroot already exists ($DATAROOT/trainA), skipping prepare."
fi

# --- 2) Train (GAN + L1 loss logged; val SSIM in validation_train.csv) ---
echo ""
echo ">>> [2/4] Training..."
python train.py \
  --dataroot "$DATAROOT" \
  --name "$NAME" \
  --model pix2pix \
  --dataset_mode aligned \
  --direction AtoB \
  --display_id 0 \
  --gpu_ids "$GPU_IDS" \
  --lr "$LR" \
  --lambda_L1 "$LAMBDA_L1" \
  --no_flip \
  --netG "$NET_G" \
  --n_epochs "$N_EPOCHS" \
  --n_epochs_decay "$N_EPOCHS_DECAY" \
  --lr_policy step \
  --batch_size "$BATCH_SIZE" \
  --loss_type L1 \
  --val_freq "$VAL_FREQ" \
  --save_epoch_freq "$SAVE_EPOCH_FREQ" \
  --load_size "$LOAD_SIZE" \
  --crop_size "$CROP_SIZE" \
  --preprocess none \
  --no_html

VAL_CSV="checkpoints/${NAME}/validation_train.csv"
if [[ -f "$VAL_CSV" ]]; then
  echo ""
  echo "--- Validation SSIM (during training) — last rows ---"
  tail -5 "$VAL_CSV"
fi
if [[ -f "checkpoints/${NAME}/loss_log.txt" ]]; then
  echo ""
  echo "--- Training loss log — last lines ---"
  tail -8 "checkpoints/${NAME}/loss_log.txt"
fi

# --- 3) Test on test set ---
FINAL_EPOCH=$((N_EPOCHS + N_EPOCHS_DECAY))
TEST_EPOCH="${TEST_EPOCH:-$FINAL_EPOCH}"
RESULTS_DIR="results/${NAME}/test_${TEST_EPOCH}"

echo ""
echo ">>> [3/4] Testing (epoch=${TEST_EPOCH})..."
python test.py \
  --dataroot "$DATAROOT" \
  --name "$NAME" \
  --model pix2pix \
  --dataset_mode aligned \
  --direction AtoB \
  --phase test \
  --epoch "$TEST_EPOCH" \
  --num_test "$NUM_TEST" \
  --eval \
  --netG "$NET_G" \
  --gpu_ids "$GPU_IDS" \
  --load_size "$LOAD_SIZE" \
  --crop_size "$CROP_SIZE" \
  --preprocess none \
  --no_flip \
  --display_id -1

if [[ ! -d "$RESULTS_DIR" ]]; then
  # fallback if only 'latest' was saved
  RESULTS_DIR="results/${NAME}/test_latest"
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "WARN: results dir not found; try: ls results/${NAME}/"
  exit 1
fi

# --- 4) Official metrics (SSIM / Pearson / PSNR per marker) ---
echo ""
echo ">>> [4/4] post_process metrics → ${RESULTS_DIR}/score.csv"
python post_process.py --srcdir "$RESULTS_DIR"

SCORE_CSV="${RESULTS_DIR}/score.csv"
if [[ -f "$SCORE_CSV" ]]; then
  echo ""
  echo "=== Test metrics summary (mean over images) ==="
  python - <<PY
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("$SCORE_CSV").open()))
if not rows:
    raise SystemExit("empty score.csv")
keys = [k for k in rows[0] if k.endswith("_ssim") or k.endswith("_pearson") or k.endswith("_psnr")]
for k in keys:
    vals = [float(r[k]) for r in rows if r.get(k)]
    if vals:
        print(f"  {k:20s}  mean={sum(vals)/len(vals):.4f}  (n={len(vals)})")
PY
  echo ""
  echo "Full per-image table: $SCORE_CSV"
fi

echo ""
echo "=============================================="
echo " Done: $(date)"
echo " Checkpoints : checkpoints/${NAME}/"
echo " Val SSIM log: ${VAL_CSV}"
echo " Test images : ${RESULTS_DIR}"
echo "=============================================="
