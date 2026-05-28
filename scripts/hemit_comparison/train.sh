#!/usr/bin/env bash
# DEPRECATED for CUT/ASP/CycleGAN — use native pipeline instead:
#   MODEL=cut|asp|cyclegan MODE=train bash scripts/run_hemit_all.sh
# This script remains for fm / fm_plus only (via run_hemit_all.sh).
set -euo pipefail
source "$(dirname "$0")/common.sh"
case "${MODEL}" in
  cut|asp|cyclegan)
    echo "ERROR: ${MODEL} is now trained via train.py (joint 3ch)." >&2
    echo "  MODEL=${MODEL} MODE=train bash scripts/run_hemit_all.sh" >&2
    exit 1
    ;;
esac
cd "${REPO_ROOT}"

DATA_ROOT="${VS_DATA_ROOT:-${HEMIT_SRC:-data/hemit}}"
MARKER="${MARKER:-DAPI}"
PATCH_SIZE="${PATCH_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-100}"
CKPT_DIR="${CKPT_DIR:-models/baselines}"

case "${MODEL}" in
  fm|vanilla_fm)
    echo "==> train flow matching (vanilla) | marker=${MARKER}"
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
  fm_plus)
    echo "==> train flow matching (fm_plus, joint 3ch)"
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
    echo "MODEL=${MODEL} — use cut|asp|cyclegan|fm|fm_plus (pix2pix uses MODEL=pix2pix + train.py)" >&2
    exit 1
    ;;
esac
