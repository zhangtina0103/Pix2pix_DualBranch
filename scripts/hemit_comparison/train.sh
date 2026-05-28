#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "${REPO_ROOT}"

DATA_ROOT="${VS_DATA_ROOT:-${HEMIT_SRC:-data/hemit}}"
MARKER="${MARKER:-DAPI}"
PATCH_SIZE="${PATCH_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-100}"
CKPT_DIR="${CKPT_DIR:-models/baselines}"

case "${MODEL}" in
  vs_pix2pix|vs_cyclegan|vs_cut|vs_asp)
    GAN="${MODEL#vs_}"
    echo "==> train ${GAN} | ${MARKER} | patch=${PATCH_SIZE}"
    python hemit/training/train_baseline.py \
      --model "${GAN}" \
      --marker "${MARKER}" \
      --data-root "${DATA_ROOT}" \
      --epochs "${EPOCHS}" \
      --patch-size "${PATCH_SIZE}" \
      --batch-size "${BATCH_SIZE}" \
      --checkpoint-dir "${CKPT_DIR}"
    ;;
  vs_vanilla_fm)
    echo "==> train vanilla_fm | ${MARKER}"
    python hemit/training/flow_matching_adapted.py \
      --dataset hemit \
      --data_root "${DATA_ROOT}" \
      --marker "${MARKER}" \
      --in_ch 3 --out_ch 1 \
      --batch_size "${BATCH_SIZE}" \
      --max_epochs "${EPOCHS:-300}" \
      --patch_size "${PATCH_SIZE}" \
      --num_workers 2 \
      --lr 3e-4 \
      --lambda_perc 0.1 \
      --channels 64,128,192 \
      --attn_levels 0,0,1 \
      --num_res_blocks 2 \
      --ode_steps 25 \
      --ode_method heun \
      --variant vanilla \
      --devices 0 \
      --no_ffl --no_cfg --no_film
    ;;
  vs_fm_plus)
    echo "==> train hemit_fm_plus (joint 3ch)"
    python hemit/training/flow_matching_hemit_plus.py \
      --data_root "${DATA_ROOT}" \
      --patch_size "${PATCH_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --max_epochs "${EPOCHS:-300}" \
      --devices 0 \
      --gradient_checkpointing \
      --no_val_images
    ;;
  *)
    echo "MODEL=${MODEL} is not a vs_* train target" >&2
    exit 1
    ;;
esac
