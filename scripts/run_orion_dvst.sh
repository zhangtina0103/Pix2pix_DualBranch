#!/usr/bin/env bash
# D-VST zero-shot on Orion-Lite @ 512² — pretrained HE2mIHC.ckpt (no Orion finetune).
#
#   MODEL=dvst MODE=prepare|test|metrics bash scripts/run_orion_all.sh
#   # or directly:
#   MODE=prepare|test|metrics bash scripts/run_orion_dvst.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export REPO_ROOT="$ROOT"

# shellcheck source=/dev/null
source "${ROOT}/scripts/orion_diffusion_env.sh"
hemit_diffusion_check_dvst

die() { echo "ERROR: $*" >&2; exit 1; }

DVST_CONFIG_DIR="${ROOT}/configs/dvst"
EVAL_YAML="${DVST_CONFIG_DIR}/infer_ORION_lite_test.yaml"

prepare() {
  local splits="${DVST_PREPARE_SPLITS:-test}"
  local eval_only="${DVST_EVAL_PREPARE_ONLY:-1}"
  [[ -d "${ORION_DATAROOT}/testA" ]] || die "missing ${ORION_DATAROOT}/testA — run prepare_orion_lite.sbatch first"
  echo "==> [dvst/orion] prepare → ${DVST_DATA_ROOT} @ ${IMAGE_SIZE}² (splits=${splits}, from trainA/trainB)"
  local prep_args=(
    --src "${ORION_DATAROOT}" --format dvst --dst "${DVST_DATA_ROOT}"
    --from-ab --resize "${IMAGE_SIZE}" --splits "${splits}"
  )
  if [[ "${eval_only}" == "1" && "${splits}" == "test" ]]; then
    prep_args+=(--eval-only)
  fi
  python scripts/prepare_hemit_for_diffusion.py "${prep_args[@]}"
  mkdir -p "${DVST_ROOT}/data"
  ln -sfn "${DVST_DATA_ROOT}" "${DVST_ROOT}/data/ORION_LITE"
  echo "Linked ${DVST_ROOT}/data/ORION_LITE → ${DVST_DATA_ROOT}"
}

test_infer() {
  echo "==> [dvst/orion] zero-shot infer (HE2mIHC.ckpt) on Orion-Lite test"
  python "${ROOT}/scripts/hemit_dvst_make_eval_config.py" \
    --data-root "${DVST_DATA_ROOT}" \
    --split test \
    --output "${EVAL_YAML}" \
    --sample-size "${IMAGE_SIZE}" \
    --video-length "${DVST_EVAL_VIDEO_LENGTH:-1}" \
    --reference-mode "${DVST_REF_MODE:-paired_gt}" \
    ${DVST_MAX_PAIRS:+--max-pairs "${DVST_MAX_PAIRS}"}
  local ckpt="${DVST_EVAL_CKPT:-${DVST_CKPT}}"
  [[ -f "${ckpt}" ]] || die "missing checkpoint: ${ckpt}"
  for req in \
    "${DVST_ROOT}/weights/dvst_pretrained/transformer/config.json" \
    "${DVST_ROOT}/weights/dvst_pretrained/vae/config.json" \
    "${DVST_ROOT}/weights/dvst_pretrained/image_encoder/config.json"; do
    [[ -f "${req}" ]] || die "missing D-VST weight: ${req}"
  done
  (
    cd "${DVST_ROOT}"
    export PYTHONPATH="${DVST_ROOT}:${PYTHONPATH:-}"
    DVST_EVAL_CKPT="${ckpt}" \
    python eval.py --config "${EVAL_YAML}"
  )
}

metrics() {
  echo "==> [dvst/orion] export predictions → post_process.py"
  [[ -d "${DVST_DATA_ROOT}/test/label" ]] || die "missing ${DVST_DATA_ROOT}/test/label — run MODE=prepare first"
  python "${ROOT}/scripts/hemit_dvst_export_metrics.py" \
    --dvst-root "${DVST_ROOT}" \
    --pix2pix-root "${ROOT}" \
    --data-root "${DVST_DATA_ROOT}" \
    --split test \
    --infer-pattern "${DVST_INFER_PATTERN}" \
    --output-dir "${DIFFUSION_RESULTS_ROOT}/dvst_zero_shot/images"
  python post_process.py --srcdir "${DIFFUSION_RESULTS_ROOT}/dvst_zero_shot/images"
  cp -f "${DIFFUSION_RESULTS_ROOT}/dvst_zero_shot/images/score.csv" \
    "${DIFFUSION_RESULTS_ROOT}/dvst_zero_shot/score.csv" 2>/dev/null || true
  echo "Score: ${DIFFUSION_RESULTS_ROOT}/dvst_zero_shot/score.csv"
}

MODE="${MODE:-prepare|test|metrics}"
run_one() {
  case "$1" in
  prepare) prepare ;;
  test) test_infer ;;
  metrics) metrics ;;
  train) die "Orion D-VST is zero-shot only — use pretrained ${DVST_CKPT}, not train." ;;
  all) die "Use MODE=prepare|test|metrics for zero-shot eval." ;;
  *) die "unknown MODE=$1" ;;
  esac
}
IFS='|' read -r -a _modes <<< "${MODE}"
for _m in "${_modes[@]}"; do run_one "${_m}"; done

echo "Done [dvst/orion zero-shot]."
