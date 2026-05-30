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
if [[ "${MODEL}" == "vanilla_fm" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/vanilla_fm_env.sh"
  vanilla_fm_verify_locked_env
fi

die() { echo "ERROR: $*" >&2; exit 1; }

# MODEL (shell) must match PY_MODEL (train.py --model). Old profiles defaulted PY_MODEL=pix2pix for everything.
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
    die "MODEL=${MODEL} needs PY_MODEL=${want}, got PY_MODEL=${PY_MODEL}.
Fix: git pull && unset PY_MODEL
Old bug trained cut/asp/cyclegan as pix2pix — retrain after pull."
  fi
}

# Fair comparison: same ResNet9 generator width as pix2pix (~11.38M in *_net_G.pth).
assert_resnet9_g() {
  case "${MODEL}" in
    pix2pix|resnet9|cut|asp|cyclegan)
      [[ "${NETG}" == "resnet_9blocks" ]] || die \
        "MODEL=${MODEL} requires NETG=resnet_9blocks (~11.38M G), got NETG=${NETG}"
      [[ "${NGF:-64}" == "64" ]] || die \
        "MODEL=${MODEL} requires NGF=64, got NGF=${NGF}"
      ;;
  esac
}

assert_py_model
assert_resnet9_g
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
  local netg_label="${NETG:-(none)}"
  assert_py_model
  echo "==> [native] MODEL=${MODEL} PY_MODEL=${PY_MODEL} netG=${netg_label} name=${TRAIN_NAME}"
  if [[ "${PY_MODEL}" == "vanilla_fm" ]]; then
    echo "    backbone=${FM_BACKBONE:-monai} loss=${FM_LOSS:-x1} channels=${FM_CHANNELS} attn=${FM_ATTN:-0,0,1} (monai: widths multiple of 32)"
    echo "    perc=${FM_LAMBDA_PERC:-0.1} FM_LAMBDA_L1=${FM_LAMBDA_L1} FM_LAMBDA_SAMPLE_L1=${FM_LAMBDA_SAMPLE_L1}"
    echo "    test ODE: ${FM_STEPS} ${FM_SAMPLE_METHOD} | val: ${FM_VAL_STEPS} steps"
    echo "    batch_size=${BATCH_SIZE} (set BATCH_SIZE=1 if OOM)"
  elif [[ "${MODEL}" == "pix2pix" || "${MODEL}" == "resnet9" || "${MODEL}" == "cut" || "${MODEL}" == "asp" || "${MODEL}" == "cyclegan" ]]; then
    echo "    netG=${NETG} ngf=${NGF:-64} (~11.38M generator; python scripts/count_hemit_g_params.py --model ${MODEL})"
    if [[ "${MODEL}" == "cyclegan" ]]; then
      echo "    batch_size=${BATCH_SIZE} (2x G + 2x D; use LAMBDA_IDENTITY=0 if still OOM)"
    fi
  fi
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
        --fm_backbone "${FM_BACKBONE:-custom}"
        --fm_loss "${FM_LOSS:-x1}"
        --fm_channels "${FM_CHANNELS:-96,192,256}"
        --fm_attn_levels "${FM_ATTN:-0,0,0}"
        --fm_num_head_channels "${FM_NUM_HEAD_CHANNELS:-32}"
        --fm_num_res_blocks "${FM_RESBLOCKS:-2}"
        --fm_up_mode "${FM_UP_MODE:-bilinear}"
        --fm_time_dist "${FM_TIME_DIST:-logit_normal}"
        --fm_lambda_perc "${FM_LAMBDA_PERC:-0.1}"
        --fm_perc_size "${FM_PERC_SIZE:-256}"
        --fm_steps "${FM_STEPS:-25}"
        --fm_sample_method "${FM_SAMPLE_METHOD:-heun}"
        --fm_cfg_dropout "${FM_CFG_DROPOUT:-0.1}"
        --fm_cfg_scale "${FM_CFG_SCALE:-1.5}"
        --fm_film_hidden "${FM_FILM_HIDDEN:-128}"
        --fm_film_where "${FM_FILM_WHERE:-decoder}"
        --fm_film_reg "${FM_FILM_REG:-0}"
        --fm_null_mode "${FM_NULL_MODE:-zero}"
        --fm_lambda_l1 "${FM_LAMBDA_L1}"
        --fm_lambda_sample_l1 "${FM_LAMBDA_SAMPLE_L1}"
        --fm_val_steps "${FM_VAL_STEPS:-8}"
        --fm_channel_weights "${FM_CHANNEL_WEIGHTS:-1,2,1}"
      )
      if [[ "${FM_USE_GAN:-0}" == "1" ]]; then
        extra+=(
          --fm_use_gan
          --fm_lambda_gan "${FM_LAMBDA_GAN:-1.0}"
          --fm_gan_sample_prob "${FM_GAN_SAMPLE_PROB:-0.5}"
          --fm_gan_sample_steps "${FM_GAN_SAMPLE_STEPS:-12}"
        )
      fi
      if [[ "${FM_LOSS:-x1}" == "x1" && "${FM_BACKBONE:-custom}" == "monai" ]]; then
        extra+=(--fm_use_tanh)
      fi
      if [[ "${FM_BACKBONE:-custom}" == "monai" && -n "${FM_CROP_SIZE:-}" ]]; then
        echo "    fm_train_crop=${FM_CROP_SIZE} (monai mid-attn needs patches)"
        extra+=(--load_size "${FM_CROP_SIZE}" --crop_size "${FM_CROP_SIZE}")
      fi
      if [[ "${FM_LAMBDA_SAMPLE_L1:-0}" != "0" && "${FM_LAMBDA_SAMPLE_L1:-0}" != "0.0" ]]; then
        extra+=(
          --fm_sample_l1_prob "${FM_SAMPLE_L1_PROB:-1.0}"
          --fm_train_sample_method "${FM_TRAIN_SAMPLE_METHOD:-heun}"
          --fm_train_sample_steps "${FM_TRAIN_SAMPLE_STEPS:-0}"
        )
      fi
      if [[ "${FM_USE_CFG:-0}" == "1" ]]; then
        extra+=(--fm_use_cfg)
      fi
      if [[ "${FM_USE_FILM:-0}" == "1" ]]; then
        extra+=(--fm_use_film)
      fi
      if [[ "${FM_USE_SEG:-0}" == "1" ]]; then
        extra+=(--fm_use_seg)
      fi
      if [[ "${FM_FLOW_PATH:-noise}" == "bridge" ]]; then
        extra+=(--fm_flow_path bridge)
        [[ -n "${FM_HE_PROJ_INIT:-}" ]] && extra+=(--fm_he_proj_init "${FM_HE_PROJ_INIT}")
        [[ "${FM_BRIDGE_X0_SIGMA:-0}" != "0" ]] && extra+=(--fm_bridge_x0_sigma "${FM_BRIDGE_X0_SIGMA}")
        [[ "${FM_BRIDGE_NOISE_PROB:-0}" != "0" ]] && extra+=(--fm_bridge_noise_prob "${FM_BRIDGE_NOISE_PROB}")
      fi
      if [[ "${FM_INIT_FROM_COND:-0}" == "1" ]]; then
        extra+=(
          --fm_init_from_cond
          --fm_init_noise_sigma "${FM_INIT_NOISE_SIGMA:-0.3}"
        )
      fi
      echo "    FM conditioning CLI: dataset_mode=${DATASET_MODE:-aligned} seg=${FM_USE_SEG:-0} flow=${FM_FLOW_PATH:-noise} init_cond=${FM_INIT_FROM_COND:-0}"
      ;;
  esac
  local train_args=(
    --dataroot "${DATAROOT}" --name "${TRAIN_NAME}"
    --model "${PY_MODEL}" --direction AtoB --display_id "${DISPLAY_ID:-0}"
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
  # vanilla_fm uses a conditional UNet (netG), not resnet_9blocks
  if [[ "${PY_MODEL}" != "vanilla_fm" ]]; then
    train_args+=(--netG "${NETG}" --ngf "${NGF:-64}")
  fi
  if [[ "${CONTINUE_TRAIN:-0}" == "1" ]]; then
    local load_epoch="${RESUME_FROM_EPOCH:?Set RESUME_FROM_EPOCH when CONTINUE_TRAIN=1}"
    local epoch_count="${EPOCH_COUNT:-$((load_epoch + 1))}"
    local end_epoch=$((N_EPOCHS + N_EPOCHS_DECAY))
    echo "==> resume: load epoch ${load_epoch}, train epochs ${epoch_count}..${end_epoch}"
    train_args+=(--continue_train --epoch "${load_epoch}" --epoch_count "${epoch_count}")
  fi
  python train.py "${train_args[@]}"
}

test_one() {
  local name="$1" epoch="$2" num_test="$3"
  assert_py_model
  echo "==> [native test] MODEL=${MODEL} PY_MODEL=${PY_MODEL} name=${name} epoch=${epoch}"
  local extra=()
  [[ -n "${DATASET_MODE:-}" ]] && extra+=(--dataset_mode "${DATASET_MODE}")
  case "${PY_MODEL}" in
    vanilla_fm)
      echo "    fm_test: backbone=${FM_BACKBONE:-custom} loss=${FM_LOSS:-x1} steps=${FM_STEPS:-25}"
      extra+=(
        --fm_backbone "${FM_BACKBONE:-custom}"
        --fm_loss "${FM_LOSS:-x1}"
        --fm_channels "${FM_CHANNELS:-64,128,192}"
        --fm_attn_levels "${FM_ATTN:-0,0,0}"
        --fm_num_head_channels "${FM_NUM_HEAD_CHANNELS:-32}"
        --fm_num_res_blocks "${FM_RESBLOCKS:-2}"
        --fm_up_mode "${FM_UP_MODE:-bilinear}"
        --fm_steps "${FM_STEPS:-25}"
        --fm_sample_method "${FM_SAMPLE_METHOD:-heun}"
        --fm_cfg_dropout "${FM_CFG_DROPOUT:-0.1}"
        --fm_cfg_scale "${FM_CFG_SCALE:-1.5}"
        --fm_film_hidden "${FM_FILM_HIDDEN:-128}"
        --fm_film_where "${FM_FILM_WHERE:-decoder}"
        --fm_film_reg "${FM_FILM_REG:-0}"
        --fm_null_mode "${FM_NULL_MODE:-zero}"
      )
      if [[ "${FM_USE_CFG:-0}" == "1" ]]; then
        extra+=(--fm_use_cfg)
      fi
      if [[ "${FM_USE_FILM:-0}" == "1" ]]; then
        extra+=(--fm_use_film)
      fi
      if [[ "${FM_USE_SEG:-0}" == "1" ]]; then
        extra+=(--fm_use_seg)
      fi
      if [[ "${FM_FLOW_PATH:-noise}" == "bridge" ]]; then
        extra+=(--fm_flow_path bridge)
        [[ -n "${FM_HE_PROJ_INIT:-}" ]] && extra+=(--fm_he_proj_init "${FM_HE_PROJ_INIT}")
        [[ "${FM_BRIDGE_X0_SIGMA:-0}" != "0" ]] && extra+=(--fm_bridge_x0_sigma "${FM_BRIDGE_X0_SIGMA}")
        [[ "${FM_BRIDGE_NOISE_PROB:-0}" != "0" ]] && extra+=(--fm_bridge_noise_prob "${FM_BRIDGE_NOISE_PROB}")
      fi
      if [[ "${FM_INIT_FROM_COND:-0}" == "1" ]]; then
        extra+=(
          --fm_init_from_cond
          --fm_init_noise_sigma "${FM_INIT_NOISE_SIGMA:-0.3}"
        )
      fi
      echo "    FM conditioning CLI: dataset_mode=${DATASET_MODE:-aligned} seg=${FM_USE_SEG:-0} flow=${FM_FLOW_PATH:-noise} init_cond=${FM_INIT_FROM_COND:-0}"
      if [[ "${FM_LOSS:-x1}" == "x1" && "${FM_BACKBONE:-custom}" == "monai" ]]; then
        extra+=(--fm_use_tanh)
      fi
      ;;
  esac
  local test_args=(
    --dataroot "${DATAROOT}" --name "${name}"
    --model "${PY_MODEL}" --direction AtoB
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
