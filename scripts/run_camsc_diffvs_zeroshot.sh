#!/usr/bin/env bash
# DiffVS zero-shot on CaMSC: HEMIT-trained checkpoint, no CaMSC training.
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   bash scripts/run_camsc_diffvs_zeroshot.sh
#
# Env:
#   DIFFVS_CHECKPOINT_DIR  — HEMIT stage-2 ckpt (default: stage2 epoch 5)
#   CAMSC_EVAL_FOLDS       — e.g. "0" or "0 1 2 3 4" (default: all folds)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export REPO_ROOT="${ROOT}"
cd "${ROOT}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/camsc_scratch_env.sh"
# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_diffusion_env.sh"
hemit_diffusion_check_diffvs

die() { echo "ERROR: $*" >&2; exit 1; }

export CAMSC_MODEL=diffvs_zeroshot
export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
export IMAGE_SIZE="${IMAGE_SIZE:-512}"
export CAMSC_DIFFVS_DATA="${CAMSC_DIFFVS_DATA:-${CAMSC_SCRATCH_ROOT}/datasets/camsc_diffvs_kfold}"
export CAMSC_DIFFVS_INFER_ROOT="${CAMSC_DIFFVS_INFER_ROOT:-${CAMSC_SCRATCH_ROOT}/eval/diffvs_zeroshot/infer}"

CKPT="${DIFFVS_CHECKPOINT_DIR:-${DIFFVS_STAGE2_DIR}/stage2-checkpoint-epoch-${DIFFVS_STAGE2_EPOCH}}"
[[ -d "${CKPT}" ]] || die "DiffVS checkpoint not found: ${CKPT}
  Set DIFFVS_CHECKPOINT_DIR to your HEMIT-trained stage-2 directory."

FOLD_LIST="${CAMSC_EVAL_FOLDS:-}"
if [[ -z "${FOLD_LIST}" ]]; then
  FOLD_LIST="$(seq 0 $((CAMSC_KFOLDS - 1)))"
fi

EVAL_EPOCH="${CAMSC_DIFFVS_EVAL_EPOCH:-80}"
need python

for FOLD in ${FOLD_LIST}; do
  FOLD_ROOT="$(camsc_fold_dataroot "${FOLD}")"
  [[ -d "${FOLD_ROOT}/testA" ]] || die "missing ${FOLD_ROOT}/testA"

  DIFFVS_ROOT_FOLD="${CAMSC_DIFFVS_DATA}/fold${FOLD}"
  INFER_OUT="${CAMSC_DIFFVS_INFER_ROOT}/fold${FOLD}"
  RESULT_NAME="$(camsc_resolved_train_name "${FOLD}")"
  RESULT_IMAGES="${ROOT}/results/${RESULT_NAME}/test_${EVAL_EPOCH}/images"

  echo "========== DiffVS zero-shot fold ${FOLD} =========="
  echo "  prepare → ${DIFFVS_ROOT_FOLD}"
  python scripts/prepare_hemit_for_diffusion.py \
    --src "${FOLD_ROOT}" \
    --dst "${DIFFVS_ROOT_FOLD}" \
    --format diffvs \
    --from-ab \
    --splits test \
    --resize "${IMAGE_SIZE}" \
    --copy

  echo "  infer ckpt=${CKPT}"
  mkdir -p "${INFER_OUT}"
  (
    cd "${DIFFVS_ROOT}"
    export DATASET_ROOT="${DIFFVS_ROOT_FOLD}"
    export CHECKPOINT_DIR="${CKPT}"
    export OUTPUT_DIR="${INFER_OUT}"
    export PRETRAINED_MODEL="${DIFFVS_PRETRAINED}"
    bash scripts/infer_hemit_diffusion_ft.sh --image_size "${IMAGE_SIZE}"
  )

  TIFF_SRC="${INFER_OUT}/pix2pix_metrics"
  [[ -d "${TIFF_SRC}" ]] || TIFF_SRC="${INFER_OUT}"
  [[ -d "${TIFF_SRC}" ]] || die "no inference TIFFs under ${INFER_OUT}"

  mkdir -p "${RESULT_IMAGES}"
  echo "  sync TIFFs → ${RESULT_IMAGES}"
  rsync -a --delete "${TIFF_SRC}/" "${RESULT_IMAGES}/" 2>/dev/null || cp -a "${TIFF_SRC}/." "${RESULT_IMAGES}/"
done

echo "Done. Run eval_camsc_kfold.py with name-prefix camsc_bf_diffvs_zeroshot_fold epoch=${EVAL_EPOCH}"
