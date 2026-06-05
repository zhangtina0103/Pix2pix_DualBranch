#!/usr/bin/env bash
# D-VST HEMIT pipeline @ 512² — uses upstream D-VST train.py / eval.py.
#
#   MODEL=dvst MODE=prepare|train|test|metrics|all bash scripts/run_hemit_all.sh
#
# Note: D-VST uses ~610M-param PixArt DiT (pretrained). Not param-matched to 11M GAN baselines.
#       Finetune from HE2mIHC.ckpt or train from PixArt weights.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export REPO_ROOT="$ROOT"

# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_diffusion_env.sh"
hemit_diffusion_check_dvst

die() { echo "ERROR: $*" >&2; exit 1; }

DVST_CONFIG_DIR="${ROOT}/configs/dvst"
STAGE1_YAML="${DVST_CONFIG_DIR}/train1_HEMIT.yaml"
STAGE2_YAML="${DVST_CONFIG_DIR}/train2_HEMIT.yaml"
EVAL_YAML="${DVST_CONFIG_DIR}/infer_HEMIT_test.yaml"

prepare() {
  echo "==> [dvst] prepare data → ${DVST_DATA_ROOT}"
  python scripts/prepare_hemit_for_diffusion.py \
    --src "${HEMIT_SRC}" --format dvst --dst "${DVST_DATA_ROOT}"
  mkdir -p "${DVST_ROOT}/data"
  ln -sfn "${DVST_DATA_ROOT}" "${DVST_ROOT}/data/HEMIT"
  echo "Linked ${DVST_ROOT}/data/HEMIT → ${DVST_DATA_ROOT}"
}

train() {
  prepare
  [[ -f "${STAGE1_YAML}" ]] || die "missing ${STAGE1_YAML}"
  echo "==> [dvst] stage-1 (full DiT) from ${DVST_CKPT:-PixArt init}"
  (
    cd "${DVST_ROOT}"
    export PYTHONPATH="${DVST_ROOT}:${PYTHONPATH:-}"
    accelerate launch --config_file ./configs/accelerate_deepspeed.yaml \
      --main_process_port "${MAIN_PROCESS_PORT:-29510}" \
      --num_processes "${NUM_PROCESSES:-1}" \
      train.py --config "${STAGE1_YAML}"
  )
  echo "==> [dvst] stage-2 (attn_adapter only) — set checkpoint_path in ${STAGE2_YAML}"
  if [[ "${DVST_SKIP_STAGE2:-0}" != "1" ]]; then
    (
      cd "${DVST_ROOT}"
      accelerate launch --config_file ./configs/accelerate_deepspeed.yaml \
        --main_process_port "${MAIN_PROCESS_PORT:-29511}" \
        --num_processes "${NUM_PROCESSES:-1}" \
        train.py --config "${STAGE2_YAML}"
    )
  fi
}

test_infer() {
  echo "==> [dvst] generate eval config + run eval.py on test split"
  python "${ROOT}/scripts/hemit_dvst_make_eval_config.py" \
    --data-root "${DVST_DATA_ROOT}" \
    --split test \
    --output "${EVAL_YAML}" \
    --sample-size "${IMAGE_SIZE}" \
    --video-length "${DVST_EVAL_VIDEO_LENGTH:-1}" \
    --reference-mode "${DVST_REF_MODE:-paired_gt}"
  local ckpt="${DVST_EVAL_CKPT:-${DVST_CKPT}}"
  [[ -f "${ckpt}" ]] || die "missing checkpoint: ${ckpt}"
  # Patch checkpoint into generated yaml via env override in Python script
  (
    cd "${DVST_ROOT}"
    DVST_EVAL_CKPT="${ckpt}" \
    accelerate launch --config_file ./configs/accelerate_deepspeed.yaml \
      --main_process_port "${MAIN_PROCESS_PORT:-29512}" \
      --num_processes "${NUM_PROCESSES:-1}" \
      eval.py --config "${EVAL_YAML}"
  )
}

metrics() {
  echo "==> [dvst] export predictions → post_process.py"
  if [[ ! -d "${HEMIT_SRC}/test/label" && ! -d "${DIFFVS_DATA_ROOT}/test/label" ]]; then
    python scripts/prepare_hemit_for_diffusion.py \
      --src "${HEMIT_SRC}" --format diffvs --dst "${DIFFVS_DATA_ROOT}"
  fi
  local gt_root="${DIFFVS_DATA_ROOT}"
  [[ -d "${HEMIT_SRC}/test/label" ]] && gt_root="${HEMIT_SRC}"
  python "${ROOT}/scripts/hemit_dvst_export_metrics.py" \
    --dvst-root "${DVST_ROOT}" \
    --pix2pix-root "${ROOT}" \
    --data-root "${gt_root}" \
    --split test \
    --output-dir "${DIFFUSION_RESULTS_ROOT}/dvst/images"
  python post_process.py --srcdir "${DIFFUSION_RESULTS_ROOT}/dvst/images"
  cp -f "${DIFFUSION_RESULTS_ROOT}/dvst/images/score.csv" \
    "${DIFFUSION_RESULTS_ROOT}/dvst/score.csv" 2>/dev/null || true
  echo "Score: ${DIFFUSION_RESULTS_ROOT}/dvst/score.csv"
}

MODE="${MODE:-all}"
run_one() {
  case "$1" in
  prepare) prepare ;;
  train) train ;;
  test) test_infer ;;
  metrics) metrics ;;
  all) die "MODE=all not recommended for dvst (long train). Use prepare|train|test|metrics." ;;
  *) die "unknown MODE=$1" ;;
  esac
}
IFS='|' read -r -a _modes <<< "${MODE}"
for _m in "${_modes[@]}"; do run_one "${_m}"; done

echo "Done [dvst]."
