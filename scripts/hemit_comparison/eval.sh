#!/usr/bin/env bash
# Metrics for CUT / ASP / CycleGAN / vanilla FM (hemit/eval).
# Pix2pix baselines: use MODEL=pix2pix|dualbranch + MODE=metrics (post_process.py).
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "${REPO_ROOT}"

DATA_ROOT="${VS_DATA_ROOT:-${HEMIT_SRC:-data/hemit}}"
BASELINE_CKPT="${BASELINE_CKPT:-models/baselines}"
FM_CKPT="${FM_CKPT:-models/vanilla}"
EVAL_PATCH_SIZE="${EVAL_PATCH_SIZE:-1024}"
FM_PATCH_SIZE="${FM_PATCH_SIZE:-512}"
FM_STRIDE="${FM_STRIDE:-256}"
FM_STEPS="${FM_STEPS:-25}"
FM_METHOD="${FM_METHOD:-heun}"

# No hemit pix2pix — native repo pix2pix uses post_process.py instead.
GAN_MODELS="${GAN_MODELS:-cyclegan cut asp}"

mkdir -p eval/hemit/baselines eval/hemit/unified

echo "==> eval GAN (${GAN_MODELS}) → eval/hemit/baselines/"
python hemit/eval/eval_baseline.py \
  --data-root "${DATA_ROOT}" \
  --checkpoint-dir "${BASELINE_CKPT}" \
  --output-dir eval/hemit/baselines \
  --models ${GAN_MODELS} \
  --patch-size "${EVAL_PATCH_SIZE}"

echo "==> eval FM + GAN → eval/hemit/unified/"
python hemit/eval/eval_all.py \
  --data_root "$(dirname "${DATA_ROOT}")" \
  --output_dir eval/hemit/unified \
  --datasets hemit \
  --models ${GAN_MODELS} vanilla_fm \
  --baseline_ckpt_dir "${BASELINE_CKPT}" \
  --fm_ckpt_dir "${FM_CKPT}" \
  --patch_size "${EVAL_PATCH_SIZE}" \
  --fm_patch_size "${FM_PATCH_SIZE}" \
  --fm_stride "${FM_STRIDE}" \
  --fm_steps "${FM_STEPS}" \
  --fm_method "${FM_METHOD}" \
  --device cuda

echo "Done:"
echo "  eval/hemit/baselines/baseline_metrics_summary.csv"
echo "  eval/hemit/unified/by_marker.csv"
