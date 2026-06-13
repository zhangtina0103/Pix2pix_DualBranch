#!/usr/bin/env bash
# DiffVS HEMIT pipeline @ 512² — uses upstream DiffVS scripts unchanged.
#
#   MODEL=diffvs MODE=prepare|train|test|metrics|all bash scripts/run_hemit_all.sh
#
# Env: DIFFVS_ROOT, HEMIT_SRC, IMAGE_SIZE=512, DIFFVS_TRAIN_BS=4
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export REPO_ROOT="$ROOT"

# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_diffusion_env.sh"
hemit_diffusion_check_diffvs

die() { echo "ERROR: $*" >&2; exit 1; }

prepare() {
  echo "==> [diffvs] prepare data → ${DIFFVS_DATA_ROOT}"
  python scripts/prepare_hemit_for_diffusion.py \
    --src "${HEMIT_SRC}" --format diffvs --dst "${DIFFVS_DATA_ROOT}"
}

train() {
  prepare
  echo "==> [diffvs] stage-1 Marigold @ ${IMAGE_SIZE}²"
  (
    cd "${DIFFVS_ROOT}"
    export DATASET_ROOT="${DIFFVS_DATA_ROOT}"
    export OUTPUT_DIR="${DIFFVS_STAGE1_DIR}"
    export PRETRAINED_MODEL="${DIFFVS_PRETRAINED}"
    export TRAIN_BATCH_SIZE="${DIFFVS_TRAIN_BS}"
    export NUM_EPOCHS="${DIFFVS_STAGE1_EPOCHS}"
    export NUM_PROCESSES="${NUM_PROCESSES:-1}"
    bash scripts/train_hemit_stage1_marigold.sh \
      --image_size "${IMAGE_SIZE}"
  )
  echo "==> [diffvs] stage-2 Diffusion-FT"
  (
    cd "${DIFFVS_ROOT}"
    export DATASET_ROOT="${DIFFVS_DATA_ROOT}"
    export STAGE1_CHECKPOINT_DIR="${DIFFVS_STAGE1_DIR}/stage1-checkpoint-epoch-${DIFFVS_STAGE1_EPOCH}"
    export OUTPUT_DIR="${DIFFVS_STAGE2_DIR}"
    export PRETRAINED_MODEL="${DIFFVS_PRETRAINED}"
    export TRAIN_BATCH_SIZE="${DIFFVS_TRAIN_BS}"
    export NUM_EPOCHS="${DIFFVS_STAGE2_EPOCHS}"
    export NUM_PROCESSES="${NUM_PROCESSES:-1}"
    bash scripts/train_hemit_stage2_diffusion_ft.sh \
      --image_size "${IMAGE_SIZE}"
  )
}

test_infer() {
  echo "==> [diffvs] infer test split"
  local ckpt="${DIFFVS_CHECKPOINT_DIR:-${DIFFVS_STAGE2_DIR}/stage2-checkpoint-epoch-${DIFFVS_STAGE2_EPOCH}}"
  [[ -d "${ckpt}" ]] || die "missing checkpoint: ${ckpt}"
  (
    cd "${DIFFVS_ROOT}"
    export DATASET_ROOT="${DIFFVS_DATA_ROOT}"
    export CHECKPOINT_DIR="${ckpt}"
    export OUTPUT_DIR="${DIFFVS_INFER_DIR}"
    export PRETRAINED_MODEL="${DIFFVS_PRETRAINED}"
    bash scripts/infer_hemit_diffusion_ft.sh \
      --image_size "${IMAGE_SIZE}"
  )
  mkdir -p "${DIFFUSION_RESULTS_ROOT}/diffvs"
  ln -sfn "${DIFFVS_INFER_DIR}" "${DIFFUSION_RESULTS_ROOT}/diffvs/inference" 2>/dev/null || true
}

metrics() {
  [[ -d "${DIFFVS_INFER_DIR}" ]] || die "run MODE=test first (${DIFFVS_INFER_DIR})"
  echo "==> [diffvs] metrics via post_process.py"
  (
    cd "${DIFFVS_ROOT}"
    export INFERENCE_DIR="${DIFFVS_INFER_DIR}"
    export PIX2PIX_ROOT="${ROOT}"
    bash scripts/eval_hemit_metrics.sh
  )
  if [[ -f "${DIFFVS_INFER_DIR}/score.csv" ]]; then
    mkdir -p "${DIFFUSION_RESULTS_ROOT}/diffvs"
    cp -f "${DIFFVS_INFER_DIR}/score.csv" "${DIFFUSION_RESULTS_ROOT}/diffvs/score.csv"
    echo "Score: ${DIFFUSION_RESULTS_ROOT}/diffvs/score.csv"
  fi
  local tiff_export="${DIFFVS_INFER_DIR}/pix2pix_metrics"
  if [[ -d "${tiff_export}" ]]; then
    mkdir -p "${DIFFUSION_RESULTS_ROOT}/diffvs/images"
    ln -sfn "${tiff_export}" "${DIFFUSION_RESULTS_ROOT}/diffvs/images" 2>/dev/null || \
      rsync -a --delete "${tiff_export}/" "${DIFFUSION_RESULTS_ROOT}/diffvs/images/" 2>/dev/null || \
      cp -a "${tiff_export}/." "${DIFFUSION_RESULTS_ROOT}/diffvs/images/"
    echo "TIFFs: ${DIFFUSION_RESULTS_ROOT}/diffvs/images"
  fi
}

MODE="${MODE:-all}"
run_one() {
  case "$1" in
  prepare) prepare ;;
  train) train ;;
  test) test_infer ;;
  metrics) metrics ;;
  all) prepare; train; test_infer; metrics ;;
  *) die "unknown MODE=$1 (prepare|train|test|metrics|all)" ;;
  esac
}
IFS='|' read -r -a _modes <<< "${MODE}"
for _m in "${_modes[@]}"; do run_one "${_m}"; done

echo "Done [diffvs]."
