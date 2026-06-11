#!/usr/bin/env bash
# Orion-Lite: train.py → test.py → post_process.py (joint 3ch, HEMIT-matched panel).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export REPO_ROOT="${ROOT}"

MODEL="${MODEL:-pix2pix}"
MODE="${MODE:-all}"
ORION_SRC="${ORION_SRC:-./data/orion/ORIONCRC_dataset_tile_20x}"
DATAROOT="${DATAROOT:-./datasets/orion_lite}"
GPU_IDS="${GPU_IDS:-0}"

source "${ROOT}/scripts/orion_model_profiles.sh"
if [[ "${MODEL}" == "vanilla_fm" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/vanilla_fm_env.sh"
  vanilla_fm_verify_locked_env
fi

die() { echo "ERROR: $*" >&2; exit 1; }

assert_py_model() {
  local want=""
  case "${MODEL}" in
    cut) want=cut ;;
    asp) want=asp ;;
    cyclegan) want=cycle_gan ;;
    vanilla_fm) want=vanilla_fm ;;
    pix2pix|resnet9) want=pix2pix ;;
    dualbranch|resnet6|unet256|unet128|unet1024|swint|swint_unet) want=pix2pix ;;
    *) return 0 ;;
  esac
  if [[ "${PY_MODEL}" != "${want}" ]]; then
    die "MODEL=${MODEL} needs PY_MODEL=${want}, got PY_MODEL=${PY_MODEL}"
  fi
}

assert_resnet9_g() {
  case "${MODEL}" in
    pix2pix|resnet9|cut|asp|cyclegan)
      [[ "${NETG}" == "resnet_9blocks" ]] || die \
        "MODEL=${MODEL} requires NETG=resnet_9blocks, got NETG=${NETG}"
      [[ "${NGF:-64}" == "64" ]] || die "MODEL=${MODEL} requires NGF=64, got NGF=${NGF}"
      ;;
  esac
}

assert_py_model
assert_resnet9_g
need() { command -v "$1" >/dev/null 2>&1 || die "missing: $1"; }
need python
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

prepare() {
  echo "==> prepare Orion-Lite → ${DATAROOT}"
  python scripts/prepare_orion_lite.py \
    --src "${ORION_SRC}" \
    --dst "${DATAROOT}" \
    --n-train "${ORION_N_TRAIN:-1500}" \
    --tile-size "${ORION_TILE_SIZE:-512}" \
    --seed "${ORION_SEED:-42}"
  source "${ROOT}/scripts/orion_model_profiles.sh"
  echo "test tiles for eval: NUM_TEST=${NUM_TEST}"
}

# shellcheck disable=SC2317
train() {
  local netg_label="${NETG:-(none)}"
  assert_py_model
  echo "==> [orion] MODEL=${MODEL} PY_MODEL=${PY_MODEL} netG=${netg_label} name=${TRAIN_NAME}"
  echo "    dataroot=${DATAROOT} train ${LOAD_SIZE:-512}→${CROP_SIZE:-512}"
  if ! python -c "import torch; assert torch.cuda.is_available()"; then
    die "CUDA not available. Submit a GPU sbatch job."
  fi
  if [[ "${MODEL}" == "cut" || "${MODEL}" == "asp" ]]; then
    if [[ "${CROP_SIZE:-512}" -ge 1024 ]] && [[ "${BATCH_SIZE}" != "1" ]]; then
      BATCH_SIZE=1
    elif [[ "${CROP_SIZE:-512}" -lt 1024 ]] && [[ "${BATCH_SIZE}" -gt 2 ]]; then
      BATCH_SIZE=2
    fi
  fi
  local extra=()
  [[ -n "${DATASET_MODE:-}" ]] && extra+=(--dataset_mode "${DATASET_MODE}")
  case "${PY_MODEL}" in
    pix2pix) extra+=(--loss_type L1) ;;
    cut)
      extra+=(--lambda_NCE "${LAMBDA_NCE:-1.0}" --nce_patches "${NCE_PATCHES:-64}")
      [[ -n "${NCE_SIZE:-}" ]] && extra+=(--nce_size "${NCE_SIZE}")
      ;;
    asp)
      extra+=(--lambda_ASP "${LAMBDA_ASP:-1.0}" --nce_patches "${NCE_PATCHES:-64}")
      [[ -n "${NCE_SIZE:-}" ]] && extra+=(--nce_size "${NCE_SIZE}")
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
        --fm_backbone "${FM_BACKBONE:-custom}"
        --fm_loss "${FM_LOSS:-x1}"
        --fm_channels "${FM_CHANNELS:-96,192,256}"
        --fm_attn_levels "${FM_ATTN:-0,0,0}"
        --fm_num_head_channels "${FM_NUM_HEAD_CHANNELS:-32}"
        --fm_num_res_blocks "${FM_RESBLOCKS:-2}"
        --fm_up_mode "${FM_UP_MODE:-conv_transpose}"
        --fm_time_dist "${FM_TIME_DIST:-logit_normal}"
        --fm_lambda_perc "${FM_LAMBDA_PERC:-0.1}"
        --fm_perc_size "${FM_PERC_SIZE:-256}"
        --fm_lambda_vel "${FM_LAMBDA_VEL:-0}"
        --fm_steps "${FM_STEPS:-25}"
        --fm_sample_method "${FM_SAMPLE_METHOD:-heun}"
        --fm_lambda_l1 "${FM_LAMBDA_L1}"
        --fm_lambda_sample_l1 "${FM_LAMBDA_SAMPLE_L1}"
        --fm_val_steps "${FM_VAL_STEPS:-8}"
        --fm_channel_weights "${FM_CHANNEL_WEIGHTS:-1,1,1}"
      )
      [[ "${FM_USE_CROSS_ATTN:-0}" == "1" ]] && extra+=(--fm_use_cross_attn --fm_cross_attn_heads "${FM_CROSS_ATTN_HEADS:-4}")
      [[ "${FM_CROSS_ATTN_DECODER:-0}" == "1" ]] && extra+=(--fm_cross_attn_decoder)
      ;;
  esac
  local train_args=(
    --dataroot "${DATAROOT}" --name "${TRAIN_NAME}"
    --model "${PY_MODEL}" --direction AtoB --display_id "${DISPLAY_ID:--1}"
    --gpu_ids "${GPU_IDS}"
    --load_size "${LOAD_SIZE:-512}" --crop_size "${CROP_SIZE:-512}"
    --preprocess "${PREPROCESS:-resize_and_crop}"
    --num_threads "${NUM_THREADS:-8}"
    --lr "${TRAIN_LR}" --no_flip --verbose
    --n_epochs "${N_EPOCHS}" --n_epochs_decay "${N_EPOCHS_DECAY}"
    --lr_policy step --batch_size "${BATCH_SIZE}"
    --lr_decay_iters "${LR_DECAY_ITERS:-50}"
    --val_freq "${VAL_FREQ:-5}"
    --save_epoch_freq "${SAVE_EPOCH_FREQ:-5}"
    "${extra[@]}"
  )
  case "${PY_MODEL}" in
    pix2pix|cut|asp) train_args+=(--lambda_L1 "${LAMBDA_L1:-100}") ;;
  esac
  if [[ "${PY_MODEL}" != "vanilla_fm" ]]; then
    train_args+=(--netG "${NETG}" --ngf "${NGF:-64}")
  fi
  if [[ "${CONTINUE_TRAIN:-0}" == "1" ]]; then
    train_args+=(--continue_train --epoch "${RESUME_FROM_EPOCH}" --epoch_count "${EPOCH_COUNT}")
  fi
  python train.py "${train_args[@]}"
}

test_one() {
  local name="$1" epoch="$2" num_test="$3"
  assert_py_model
  echo "==> [orion test] name=${name} epoch=${epoch} num_test=${num_test}"
  local extra=()
  [[ -n "${DATASET_MODE:-}" ]] && extra+=(--dataset_mode "${DATASET_MODE}")
  if [[ "${PY_MODEL}" == "vanilla_fm" ]]; then
    extra+=(
      --fm_backbone "${FM_BACKBONE:-custom}"
      --fm_loss "${FM_LOSS:-x1}"
      --fm_channels "${FM_CHANNELS:-96,192,256}"
      --fm_attn_levels "${FM_ATTN:-0,0,0}"
      --fm_steps "${FM_STEPS:-25}"
      --fm_sample_method "${FM_SAMPLE_METHOD:-heun}"
    )
    [[ "${FM_USE_CROSS_ATTN:-0}" == "1" ]] && extra+=(--fm_use_cross_attn --fm_cross_attn_heads "${FM_CROSS_ATTN_HEADS:-4}")
    [[ "${FM_CROSS_ATTN_DECODER:-0}" == "1" ]] && extra+=(--fm_cross_attn_decoder)
  fi
  local test_args=(
    --dataroot "${DATAROOT}" --name "${name}"
    --model "${PY_MODEL}" --direction AtoB
    --gpu_ids "${GPU_IDS}"
    --load_size "${LOAD_SIZE:-512}" --crop_size "${CROP_SIZE:-512}"
    --preprocess "${PREPROCESS:-resize_and_crop}"
    --epoch "${epoch}" --num_test "${num_test}" --eval --verbose
    "${extra[@]}"
  )
  if [[ "${PY_MODEL}" != "vanilla_fm" ]]; then
    test_args+=(--netG "${NETG}" --ngf "${NGF:-64}")
  fi
  python test.py "${test_args[@]}"
}

metrics_one() {
  local name="$1" epoch="$2"
  [[ -d "results/${name}/test_${epoch}/images" ]] || die "run MODE=test first"
  python post_process.py --srcdir "results/${name}/test_${epoch}/"
}

run_mode() {
  case "$1" in
    prepare) prepare ;;
    train) train ;;
    test) test_one "${TRAIN_NAME}" "${TEST_EPOCH}" "${NUM_TEST}" ;;
    metrics) metrics_one "${TRAIN_NAME}" "${TEST_EPOCH}" ;;
    all)
      prepare
      train
      test_one "${TRAIN_NAME}" "${TEST_EPOCH}" "${NUM_TEST}"
      metrics_one "${TRAIN_NAME}" "${TEST_EPOCH}"
      ;;
    *) die "unknown MODE='$1'" ;;
  esac
}

IFS='|' read -r -a _modes <<< "${MODE}"
for m in "${_modes[@]}"; do run_mode "${m}"; done
