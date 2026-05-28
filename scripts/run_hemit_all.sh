#!/usr/bin/env bash
# Unified HEMIT pipeline — same steps for every model family.
#
#   prepare → train → test → metrics
#
# Native (this repo train.py / test.py / post_process.py, joint 3ch):
#   MODEL=dualbranch | pix2pix_resnet9 | pix2pix_unet256
#
# Comparison (hemit/ in this repo — per-marker GAN + FM):
#   MODEL=vs_pix2pix | vs_cyclegan | vs_cut | vs_asp | vs_vanilla_fm | vs_fm_plus
#   MODEL=vs_eval          # eval all comparison models (no train/test in this repo)
#
# Examples:
#   MODEL=dualbranch MODE=all bash scripts/run_hemit_all.sh
#   MODEL=dualbranch MODE=test|metrics TEST_EPOCH=80 bash scripts/run_hemit_all.sh
#   MODEL=vs_pix2pix MARKER=CD3 MODE=train bash scripts/run_hemit_all.sh
#   MODEL=vs_eval MODE=metrics bash scripts/run_hemit_all.sh
#   MODEL=vs_baselines_hint MODE=train  # print sbatch for 12 baseline jobs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export MODEL="${MODEL:-dualbranch}"
export MODE="${MODE:-all}"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export GPU_IDS="${GPU_IDS:-0}"
export VS_DATA_ROOT="${VS_DATA_ROOT:-${HEMIT_SRC}}"

die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing: $1"; }
need python
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

is_vs_model() { [[ "${MODEL}" == vs_* ]]; }

prepare() {
  echo "==> prepare pix2pix dataroot: ${DATAROOT}"
  [[ -d "${HEMIT_SRC}/train" ]] || die "HEMIT_SRC not found: ${HEMIT_SRC}"
  local copy_flag=()
  [[ "${PREPARE_COPY:-0}" == "1" ]] && copy_flag=(--copy)
  python scripts/prepare_hemit_data.py \
    --src "${HEMIT_SRC}" \
    --dst "${DATAROOT}" \
    "${copy_flag[@]}"
}

prepare_vs_note() {
  echo "==> prepare: vs_* models use native HEMIT at VS_DATA_ROOT=${VS_DATA_ROOT}"
  echo "    (no trainA/B needed; skip or run prepare for dualbranch / pix2pix_* only)"
}

# --- native pix2pix stack (dualbranch, resnet9, unet) ---
native_train() {
  # shellcheck source=scripts/hemit_model_profiles.sh
  source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"
  echo "==> train MODEL=${MODEL} netG=${NETG} name=${TRAIN_NAME}"
  python train.py \
    --dataroot "${DATAROOT}" \
    --name "${TRAIN_NAME}" \
    --model pix2pix \
    --direction AtoB \
    --display_id 0 \
    --lr "${TRAIN_LR}" \
    --lambda_L1 "${LAMBDA_L1}" \
    --no_flip \
    --netG "${NETG}" \
    --n_epochs "${N_EPOCHS}" \
    --n_epochs_decay "${N_EPOCHS_DECAY}" \
    --lr_policy step \
    --batch_size "${BATCH_SIZE}" \
    --loss_type L1 \
    --val_freq 5
}

native_test() {
  source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"
  echo "==> test name=${TRAIN_NAME} epoch=${TEST_EPOCH} netG=${NETG}"
  python test.py \
    --dataroot "${DATAROOT}" \
    --name "${TRAIN_NAME}" \
    --model pix2pix \
    --direction AtoB \
    --epoch "${TEST_EPOCH}" \
    --num_test "${NUM_TEST}" \
    --eval \
    --netG "${NETG}"
}

native_metrics() {
  source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"
  local srcdir="results/${TRAIN_NAME}/test_${TEST_EPOCH}/images"
  echo "==> metrics (post_process) ${srcdir}"
  [[ -d "${srcdir}" ]] || die "missing ${srcdir} — run MODE=test first"
  python post_process.py --srcdir "results/${TRAIN_NAME}/test_${TEST_EPOCH}/" \
    ${METRICS_COMPOSITE:+--composite}
}

native_eval_pretrained() {
  source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"
  [[ -d "checkpoints/${PRETRAINED_NAME}" ]] || die "missing checkpoints/${PRETRAINED_NAME}"
  TEST_EPOCH="${PRETRAINED_EPOCH}" TRAIN_NAME="${PRETRAINED_NAME}" native_test
  TEST_EPOCH="${PRETRAINED_EPOCH}" TRAIN_NAME="${PRETRAINED_NAME}" native_metrics
}

vs_train() {
  export MODEL REPO_ROOT VS_DATA_ROOT MARKER PATCH_SIZE BATCH_SIZE EPOCHS CKPT_DIR
  bash "${REPO_ROOT}/scripts/hemit_comparison/train.sh"
}

vs_metrics() {
  bash "${REPO_ROOT}/scripts/hemit_comparison/eval.sh"
}

vs_baselines_hint() {
  cat <<EOF
==> Train 12 GAN baselines (submit one job per model/marker or loop):

  for m in pix2pix cyclegan cut asp; do
    for mk in DAPI panCK CD3; do
      MODEL=vs_\${m} MARKER=\${mk} MODE=train bash scripts/run_hemit_all.sh
    done
  done

EOF
}

vs_fm_hint() {
  cat <<EOF
==> Train FM (in-repo hemit/training/)

  MODEL=vs_vanilla_fm MARKER=DAPI MODE=train bash scripts/run_hemit_all.sh
  MODEL=vs_fm_plus MODE=train bash scripts/run_hemit_all.sh

EOF
}

run_mode() {
  case "$1" in
    prepare)
      if is_vs_model; then prepare_vs_note
      else prepare
      fi
      ;;
    train)
      case "${MODEL}" in
        vs_baselines_hint) vs_baselines_hint ;;
        vs_fm_hint) vs_fm_hint ;;
        vs_eval) die "MODEL=vs_eval has no train step; use MODE=metrics" ;;
        vs_*) vs_train ;;
        *) native_train ;;
      esac
      ;;
    test)
      is_vs_model && die "vs_* models: inference is inside eval_all.py — use MODE=metrics with MODEL=vs_eval"
      native_test
      ;;
    metrics)
      if [[ "${MODEL}" == vs_eval ]] || [[ "${MODEL}" == vs_* && "${VS_METRICS_ONLY:-0}" == 1 ]]; then
        vs_metrics
      elif is_vs_model; then
        vs_metrics
      else
        native_metrics
      fi
      ;;
    eval_pretrained)
      is_vs_model && die "no eval_pretrained for vs_* models"
      native_eval_pretrained
      ;;
    all)
      if is_vs_model; then
        die "MODE=all not defined for vs_* — use MODE=train then MODE=metrics, or SLURM arrays"
      fi
      prepare
      native_train
      native_test
      native_metrics
      ;;
    *)
      die "unknown MODE='$1'"
      ;;
  esac
}

echo "===== HEMIT pipeline MODEL=${MODEL} MODE=${MODE} ====="
IFS='|' read -r -a _modes <<< "${MODE}"
for m in "${_modes[@]}"; do
  run_mode "${m}"
done
echo "Done."
