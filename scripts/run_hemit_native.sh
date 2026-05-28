#!/usr/bin/env bash
# This repo's pix2pix: train.py → test.py → post_process.py (joint 3-channel).
# Called by run_hemit_all.sh for MODEL=dualbranch | pix2pix | resnet9 | ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-dualbranch}"
MODE="${MODE:-all}"
HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
DATAROOT="${DATAROOT:-./datasets/hemit}"
GPU_IDS="${GPU_IDS:-0}"

source "${ROOT}/scripts/hemit_model_profiles.sh"

die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing: $1"; }
need python
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

prepare() {
  echo "==> prepare ${DATAROOT}"
  [[ -d "${HEMIT_SRC}/train" ]] || die "HEMIT_SRC not found: ${HEMIT_SRC}"
  local copy_flag=()
  [[ "${PREPARE_COPY:-0}" == "1" ]] && copy_flag=(--copy)
  python scripts/prepare_hemit_data.py \
    --src "${HEMIT_SRC}" --dst "${DATAROOT}" "${copy_flag[@]}"
}

train() {
  echo "==> [native] model=${PY_MODEL} netG=${NETG} name=${TRAIN_NAME}"
  if ! python -c "import torch; assert torch.cuda.is_available()"; then
    die "CUDA not available (login node?). Submit a GPU job, e.g.:
  MODEL=cut MODE=train sbatch bash_scripts/run_hemit_all.sbatch
  # or: srun --partition=mit_normal_gpu --gres=gpu:1 --mem=64G --time=06:00:00 --pty bash"
  fi
  python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
  local extra=()
  [[ -n "${DATASET_MODE:-}" ]] && extra+=(--dataset_mode "${DATASET_MODE}")
  case "${PY_MODEL}" in
    pix2pix)
      extra+=(--loss_type L1)
      ;;
    cut)
      extra+=(--lambda_NCE "${LAMBDA_NCE:-1.0}")
      ;;
    asp)
      extra+=(--lambda_ASP "${LAMBDA_ASP:-1.0}")
      ;;
    cycle_gan)
      extra+=(
        --lambda_A "${LAMBDA_A:-10.0}"
        --lambda_B "${LAMBDA_B:-10.0}"
        --lambda_identity "${LAMBDA_IDENTITY:-0.5}"
      )
      ;;
    vanilla_fm)
      extra+=(
        --fm_channels "${FM_CHANNELS:-32,64,96}"
        --fm_num_res_blocks "${FM_RESBLOCKS:-1}"
        --fm_steps "${FM_STEPS:-25}"
      )
      ;;
  esac
  python train.py \
    --dataroot "${DATAROOT}" --name "${TRAIN_NAME}" \
    --model "${PY_MODEL}" --direction AtoB --display_id 0 \
    --lr "${TRAIN_LR}" --no_flip --netG "${NETG}" \
    --n_epochs "${N_EPOCHS}" --n_epochs_decay "${N_EPOCHS_DECAY}" \
    --lr_policy step --batch_size "${BATCH_SIZE}" \
    --val_freq 5 \
    --lambda_L1 "${LAMBDA_L1:-100}" \
    "${extra[@]}"
}

test_one() {
  local name="$1" epoch="$2" num_test="$3"
  local extra=()
  [[ -n "${DATASET_MODE:-}" ]] && extra+=(--dataset_mode "${DATASET_MODE}")
  case "${PY_MODEL}" in
    vanilla_fm)
      extra+=(--fm_channels "${FM_CHANNELS:-32,64,96}" --fm_num_res_blocks "${FM_RESBLOCKS:-1}" --fm_steps "${FM_STEPS:-25}")
      ;;
  esac
  python test.py \
    --dataroot "${DATAROOT}" --name "${name}" \
    --model "${PY_MODEL}" --direction AtoB \
    --epoch "${epoch}" --num_test "${num_test}" --eval --netG "${NETG}" \
    "${extra[@]}"
}

metrics_one() {
  local name="$1" epoch="$2"
  [[ -d "results/${name}/test_${epoch}/images" ]] || die "run MODE=test first"
  python post_process.py --srcdir "results/${name}/test_${epoch}/" \
    ${METRICS_COMPOSITE:+--composite}
}

eval_pretrained() {
  [[ -d "checkpoints/${PRETRAINED_NAME}" ]] || die "missing checkpoints/${PRETRAINED_NAME}"
  test_one "${PRETRAINED_NAME}" "${PRETRAINED_EPOCH}" "${NUM_TEST}"
  metrics_one "${PRETRAINED_NAME}" "${PRETRAINED_EPOCH}"
}

run_mode() {
  case "$1" in
    prepare) prepare ;;
    train) train ;;
    test) test_one "${TRAIN_NAME}" "${TEST_EPOCH}" "${NUM_TEST}" ;;
    metrics) metrics_one "${TRAIN_NAME}" "${TEST_EPOCH}" ;;
    eval_pretrained) eval_pretrained ;;
    all) prepare; train; test_one "${TRAIN_NAME}" "${TEST_EPOCH}" "${NUM_TEST}"; metrics_one "${TRAIN_NAME}" "${TEST_EPOCH}" ;;
    *) die "unknown MODE='$1'" ;;
  esac
}

IFS='|' read -r -a _modes <<< "${MODE}"
for m in "${_modes[@]}"; do run_mode "${m}"; done
