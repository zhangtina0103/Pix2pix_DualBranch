# shellcheck shell=bash
# Pin vanilla_fm training env. Source from run_hemit_vanilla_fm.sh / train sbatch.
#
# Mentor integration (flow_matching.py + flow_matching_v.py via MONAI UNet):
#   FM_LOSS=x1     → L1 + perceptual, logit-normal t, tanh, x1→v Heun ODE
#   FM_LOSS=velocity → velocity MSE, uniform t, raw-v Heun ODE
# Param match ResNet9 (~11.38M): FM_CHANNELS=64,128,192 FM_RESBLOCKS=2
# 1024² train: FM_ATTN=0,0,0 (attention OOMs on L40S; mentor used 128² crops)
# MONAI UNet: each channel width must be divisible by 32 (not 272 — use 256)
# Tune on login node: python scripts/count_fm_params.py --search

# Remove flags from a prior run / login shell (sbatch without --export=ALL still inherits some shells).
vanilla_fm_clear_stale_env() {
  unset FM_USE_CFG FM_USE_FILM FM_USE_GAN FM_USE_ODE_TRAIN FM_CROP_SIZE
  unset FM_NULL_MODE FM_FILM_WHERE FM_FILM_REG
  unset FM_LAMBDA_SAMPLE_L1 FM_LAMBDA_L1 FM_USE_TANH
  unset FM_USE_SEG FM_FLOW_PATH FM_INIT_FROM_COND FM_INIT_NOISE_SIGMA DATASET_MODE
  unset VANILLA_FM_COND_PROFILE
}

vanilla_fm_verify_cond_profile() {
  [[ "${VANILLA_FM_ENV_LOCKED:-0}" == "1" ]] || return 0
  local profile="${VANILLA_FM_COND_PROFILE:-}"
  [[ -n "${profile}" ]] || return 0

  _fm_cond_fail() {
    echo "ERROR: cond profile=${profile} check failed: $*" >&2
    echo "  TRAIN_NAME=${TRAIN_NAME:-} FM_USE_SEG=${FM_USE_SEG:-} FM_FLOW_PATH=${FM_FLOW_PATH:-}" >&2
    echo "  FM_INIT_FROM_COND=${FM_INIT_FROM_COND:-} DATASET_MODE=${DATASET_MODE:-}" >&2
    echo "  FM_USE_CFG=${FM_USE_CFG:-} FM_USE_FILM=${FM_USE_FILM:-} FM_STEPS=${FM_STEPS:-}" >&2
    exit 1
  }

  [[ "${FM_USE_CFG:-0}" == "0" ]] || _fm_cond_fail "FM_USE_CFG must be 0 for conditioning runs"
  [[ "${FM_USE_FILM:-0}" == "0" ]] || _fm_cond_fail "FM_USE_FILM must be 0 for conditioning runs"
  [[ "${FM_USE_GAN:-0}" == "0" ]] || _fm_cond_fail "FM_USE_GAN must be 0 for conditioning runs"
  [[ "${FM_STEPS:-}" == "25" ]] || _fm_cond_fail "FM_STEPS must be 25 (got ${FM_STEPS:-unset})"
  [[ "${FM_CHANNEL_WEIGHTS:-}" == "1,1,1" ]] || _fm_cond_fail "FM_CHANNEL_WEIGHTS must be 1,1,1"

  case "${profile}" in
    seg)
      [[ "${FM_USE_SEG:-0}" == "1" ]] || _fm_cond_fail "FM_USE_SEG=1 required"
      [[ "${DATASET_MODE:-}" == "aligned_cond" ]] || _fm_cond_fail "DATASET_MODE=aligned_cond required"
      [[ "${FM_FLOW_PATH:-noise}" == "noise" ]] || _fm_cond_fail "FM_FLOW_PATH must be noise"
      [[ "${FM_INIT_FROM_COND:-0}" == "0" ]] || _fm_cond_fail "FM_INIT_FROM_COND must be 0"
      ;;
    bridge)
      [[ "${FM_USE_SEG:-0}" == "0" ]] || _fm_cond_fail "FM_USE_SEG must be 0"
      [[ "${FM_FLOW_PATH:-}" == "bridge" ]] || _fm_cond_fail "FM_FLOW_PATH=bridge required"
      [[ "${FM_INIT_FROM_COND:-0}" == "0" ]] || _fm_cond_fail "FM_INIT_FROM_COND must be 0"
      [[ "${DATASET_MODE:-aligned}" == "aligned" ]] || _fm_cond_fail "DATASET_MODE must be aligned (or unset)"
      [[ "${FM_HE_PROJ_INIT:-gray}" == "gray" ]] || _fm_cond_fail "FM_HE_PROJ_INIT must be gray (v2)"
      ;;
    init_cond)
      [[ "${FM_USE_SEG:-0}" == "0" ]] || _fm_cond_fail "FM_USE_SEG must be 0"
      [[ "${FM_FLOW_PATH:-noise}" == "noise" ]] || _fm_cond_fail "FM_FLOW_PATH must be noise"
      [[ "${FM_INIT_FROM_COND:-0}" == "1" ]] || _fm_cond_fail "FM_INIT_FROM_COND=1 required"
      [[ "${DATASET_MODE:-aligned}" == "aligned" ]] || _fm_cond_fail "DATASET_MODE must be aligned (or unset)"
      ;;
    *)
      _fm_cond_fail "unknown VANILLA_FM_COND_PROFILE=${profile}"
      ;;
  esac
}

vanilla_fm_verify_locked_env() {
  [[ "${VANILLA_FM_ENV_LOCKED:-0}" == "1" ]] || return 0
  if [[ -n "${VANILLA_FM_EXPECTED_TRAIN_NAME:-}" && "${TRAIN_NAME:-}" != "${VANILLA_FM_EXPECTED_TRAIN_NAME}" ]]; then
    echo "ERROR: TRAIN_NAME=${TRAIN_NAME:-<unset>} but locked profile expects ${VANILLA_FM_EXPECTED_TRAIN_NAME}" >&2
    exit 1
  fi
  if [[ "${FM_BACKBONE:-}" != "custom" ]]; then
    echo "ERROR: FM_BACKBONE=${FM_BACKBONE:-<unset>} (locked HEMIT phases require custom)" >&2
    exit 1
  fi
  vanilla_fm_verify_cond_profile
}

vanilla_fm_apply_train_env() {
  # monai-generative: middle block ALWAYS has attention → OOM at 1024² on L40S.
  # HEMIT fair compare @ 1024: use custom (skip U-Net). monai: set FM_CROP_SIZE=512.
  export FM_BACKBONE="${FM_BACKBONE:-custom}"
  export FM_LOSS="${FM_LOSS:-x1}"
  export FM_CHANNELS="${FM_CHANNELS:-64,128,192}"
  # Attention at 1024² can request 100+ GiB; default off for HEMIT native train.
  export FM_ATTN="${FM_ATTN:-0,0,0}"
  export BATCH_SIZE="${BATCH_SIZE:-1}"
  export FM_RESBLOCKS="${FM_RESBLOCKS:-2}"
  export FM_NUM_HEAD_CHANNELS="${FM_NUM_HEAD_CHANNELS:-32}"
  export FM_LAMBDA_PERC="${FM_LAMBDA_PERC:-0.1}"
  export FM_PERC_SIZE="${FM_PERC_SIZE:-256}"
  export FM_TIME_DIST="${FM_TIME_DIST:-logit_normal}"
  export FM_STEPS="${FM_STEPS:-25}"
  export FM_VAL_STEPS="${FM_VAL_STEPS:-8}"
  export FM_SAMPLE_METHOD="${FM_SAMPLE_METHOD:-heun}"
  # custom U-Net decoder: conv_transpose = joint_perc / perc_strong; bilinear = decoder_only
  export FM_UP_MODE="${FM_UP_MODE:-conv_transpose}"

  if [[ "${FM_USE_ODE_TRAIN:-0}" == "1" ]]; then
    export FM_LAMBDA_L1="${FM_LAMBDA_L1:-10}"
    export FM_LAMBDA_SAMPLE_L1="${FM_LAMBDA_SAMPLE_L1:-100}"
    export FM_SAMPLE_L1_PROB="${FM_SAMPLE_L1_PROB:-1.0}"
    export FM_TRAIN_SAMPLE_METHOD="${FM_TRAIN_SAMPLE_METHOD:-euler}"
    echo "WARNING: FM_USE_ODE_TRAIN=1 — ODE+L1 in training loop (not standard FM)" >&2
  else
    # Force zeros: survives `sbatch --export=ALL` with stale FM_LAMBDA_SAMPLE_L1=100.
    export FM_LAMBDA_L1=0
    export FM_LAMBDA_SAMPLE_L1=0
  fi
}

# Ablation 1: same recipe as hemit_vanilla_fm_joint_perc, only bilinear decoder (after git pull).
vanilla_fm_apply_decoder_only_env() {
  vanilla_fm_apply_train_env
  export TRAIN_NAME=hemit_vanilla_fm_decoder_only
  export FM_UP_MODE=bilinear
  export FM_CHANNELS=96,192,256
  export FM_LAMBDA_PERC=0.1
  export FM_PERC_SIZE=256
  export FM_STEPS="${FM_STEPS:-25}"
  export FM_VAL_STEPS="${FM_VAL_STEPS:-8}"
  export FM_SAMPLE_METHOD="${FM_SAMPLE_METHOD:-heun}"
  export FM_TIME_DIST="${FM_TIME_DIST:-logit_normal}"
  export FM_USE_GAN=0
  export FM_CHANNEL_WEIGHTS=1,1,1
  export VAL_FREQ="${VAL_FREQ:-10}"
  echo "decoder_only: joint_perc recipe + bilinear upsample (no ODE-L1, no GAN)" >&2
}

# Ablation 2 (parallel): stronger perceptual only — still x1 + logit-normal + noise→ODE @ 25 Heun.
vanilla_fm_apply_perc_strong_env() {
  vanilla_fm_apply_train_env
  export TRAIN_NAME=hemit_vanilla_fm_perc_strong
  export FM_CHANNELS=96,192,256
  export FM_LAMBDA_PERC=0.5
  export FM_PERC_SIZE=256
  export FM_STEPS="${FM_STEPS:-25}"
  export FM_VAL_STEPS="${FM_VAL_STEPS:-8}"
  export FM_SAMPLE_METHOD="${FM_SAMPLE_METHOD:-heun}"
  export FM_TIME_DIST="${FM_TIME_DIST:-logit_normal}"
  export FM_USE_GAN=0
  export FM_CHANNEL_WEIGHTS=1,1,1
  export VAL_FREQ="${VAL_FREQ:-10}"
  echo "perc_strong: vanilla FM, perc=${FM_LAMBDA_PERC} (no ODE-L1, no GAN)" >&2
}

# Ablation 3: emphasize CD3 in x1 L1 (1,2,1) — vanilla FM, same recipe as joint_perc otherwise.
vanilla_fm_apply_cd3_weight_env() {
  vanilla_fm_clear_stale_env
  unset FM_STEPS FM_CHANNELS TRAIN_NAME
  vanilla_fm_apply_train_env
  export TRAIN_NAME=hemit_vanilla_fm_cd3_weight
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
  export FM_BACKBONE=custom
  export FM_UP_MODE=conv_transpose
  export FM_CHANNELS=96,192,256
  export FM_LAMBDA_PERC=0.1
  export FM_PERC_SIZE=256
  export FM_STEPS=25
  export FM_VAL_STEPS=8
  export FM_SAMPLE_METHOD=heun
  export FM_TIME_DIST=logit_normal
  export FM_LAMBDA_L1=0
  export FM_LAMBDA_SAMPLE_L1=0
  export FM_USE_GAN=0
  unset FM_USE_CFG FM_USE_FILM
  export FM_CHANNEL_WEIGHTS=1,2,1
  export VAL_FREQ="${VAL_FREQ:-10}"
  echo "cd3_weight: vanilla FM, channel L1 weights=${FM_CHANNEL_WEIGHTS}" >&2
}

# Mentor-style vanilla FM only: Gaussian x0, x1 L1 + perceptual, logit-normal t, Heun ODE.
# No GAN, no pix2pix-style direct G(A). Val ODE steps = test steps.
vanilla_fm_apply_mentor_strict_env() {
  vanilla_fm_apply_train_env
  export TRAIN_NAME=hemit_vanilla_fm_mentor
  export FM_CHANNELS="${FM_CHANNELS:-96,192,256}"
  export FM_LAMBDA_PERC="${FM_LAMBDA_PERC:-0.1}"
  export FM_PERC_SIZE="${FM_PERC_SIZE:-256}"
  export FM_STEPS="${FM_STEPS:-25}"
  export FM_VAL_STEPS="${FM_VAL_STEPS:-${FM_STEPS}}"
  export FM_SAMPLE_METHOD="${FM_SAMPLE_METHOD:-heun}"
  export FM_TIME_DIST="${FM_TIME_DIST:-logit_normal}"
  export FM_USE_GAN=0
  export VAL_FREQ="${VAL_FREQ:-10}"
  echo "mentor_strict: flow matching only (noise→ODE), perc=${FM_LAMBDA_PERC} val_steps=${FM_VAL_STEPS}" >&2
}

# Tune vanilla FM toward pix2pix SSIM without changing the generative path (still noise→ODE).
# GAN is OFF by default; set FM_USE_GAN=1 only if you explicitly want PatchGAN on ODE samples.
vanilla_fm_apply_beat_pix2pix_env() {
  vanilla_fm_apply_mentor_strict_env
  export TRAIN_NAME=hemit_vanilla_fm_beat_p2p
  export FM_LAMBDA_PERC="${FM_LAMBDA_PERC:-1.0}"
  export FM_PERC_SIZE="${FM_PERC_SIZE:-512}"
  export FM_STEPS="${FM_STEPS:-50}"
  export FM_VAL_STEPS="${FM_VAL_STEPS:-${FM_STEPS}}"
  export FM_USE_ODE_TRAIN=1
  export FM_LAMBDA_SAMPLE_L1="${FM_LAMBDA_SAMPLE_L1:-100}"
  export FM_SAMPLE_L1_PROB="${FM_SAMPLE_L1_PROB:-0.5}"
  export FM_TRAIN_SAMPLE_STEPS="${FM_TRAIN_SAMPLE_STEPS:-25}"
  export FM_TRAIN_SAMPLE_METHOD="${FM_TRAIN_SAMPLE_METHOD:-heun}"
  export FM_GAN_SAMPLE_STEPS="${FM_GAN_SAMPLE_STEPS:-12}"
  export FM_USE_GAN="${FM_USE_GAN:-0}"
  export FM_LAMBDA_GAN="${FM_LAMBDA_GAN:-1.0}"
  export FM_GAN_SAMPLE_PROB="${FM_GAN_SAMPLE_PROB:-0.5}"
  export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,2,1}"
  export VAL_FREQ="${VAL_FREQ:-5}"
  echo "beat_pix2pix (still FM): sample_L1=${FM_LAMBDA_SAMPLE_L1} GAN=${FM_USE_GAN} steps=${FM_STEPS}" >&2
}

# joint_perc + CFG: finetune from hemit_vanilla_fm_joint_perc/80 with cond dropout; test with fm_cfg_scale.
# joint_perc + FiLM decoder modulation (finetune from joint_perc/80; no CFG).
vanilla_fm_apply_joint_film_env() {
  vanilla_fm_clear_stale_env
  unset FM_STEPS TRAIN_NAME
  vanilla_fm_apply_train_env
  export TRAIN_NAME=hemit_vanilla_fm_joint_film
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
  export FM_BACKBONE=custom
  export FM_UP_MODE=conv_transpose
  export FM_CHANNELS=96,192,256
  export FM_LAMBDA_PERC=0.1
  export FM_CHANNEL_WEIGHTS=1,1,1
  export FM_LAMBDA_L1=0
  export FM_LAMBDA_SAMPLE_L1=0
  unset FM_USE_CFG FM_USE_GAN
  export FM_USE_FILM=1
  export FM_FILM_WHERE=decoder
  export FM_FILM_HIDDEN=128
  export FM_FILM_REG=0
  export FM_STEPS=25
  export FM_VAL_STEPS=8
  export FM_SAMPLE_METHOD=heun
  echo "joint_film: joint_perc + FiLM on decoder (no CFG)" >&2
}

vanilla_fm_apply_joint_cfg_env() {
  vanilla_fm_clear_stale_env
  unset FM_STEPS TRAIN_NAME
  vanilla_fm_apply_train_env
  export TRAIN_NAME=hemit_vanilla_fm_joint_cfg
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
  export FM_BACKBONE=custom
  export FM_UP_MODE=conv_transpose
  export FM_CHANNELS=96,192,256
  export FM_LAMBDA_PERC=0.1
  export FM_CHANNEL_WEIGHTS=1,1,1
  export FM_LAMBDA_L1=0
  export FM_LAMBDA_SAMPLE_L1=0
  unset FM_USE_FILM FM_USE_GAN
  export FM_USE_CFG=1
  export FM_NULL_MODE=zero
  export FM_CFG_DROPOUT=0.1
  export FM_CFG_SCALE=1.5
  export FM_STEPS=25
  export FM_VAL_STEPS=8
  export FM_SAMPLE_METHOD=heun
  echo "joint_cfg: joint_perc + CFG dropout=${FM_CFG_DROPOUT} test_scale=${FM_CFG_SCALE}" >&2
}

# Shared joint_perc recipe (baseline checkpoint hemit_vanilla_fm_joint_perc/80).
vanilla_fm_apply_joint_perc_pins() {
  vanilla_fm_clear_stale_env
  unset FM_STEPS FM_CHANNELS TRAIN_NAME PRETRAINED_NAME
  vanilla_fm_apply_train_env
  export FM_BACKBONE=custom
  export FM_UP_MODE=conv_transpose
  export FM_CHANNELS=96,192,256
  export FM_LAMBDA_PERC=0.1
  export FM_CHANNEL_WEIGHTS=1,1,1
  export FM_STEPS=25
  export FM_VAL_STEPS=8
  export FM_SAMPLE_METHOD=heun
  export FM_TIME_DIST=logit_normal
  export FM_LAMBDA_L1=0
  export FM_LAMBDA_SAMPLE_L1=0
  unset FM_USE_CFG FM_USE_FILM FM_USE_GAN FM_USE_ODE_TRAIN
  unset FM_NULL_MODE FM_FILM_WHERE
  export FM_FILM_REG=0
}

# Phase 1: FM+GAN — PatchGAN on 12-step ODE samples, finetune from joint_perc/80.
vanilla_fm_apply_joint_gan_env() {
  vanilla_fm_apply_joint_perc_pins
  export TRAIN_NAME=hemit_vanilla_fm_joint_gan
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
  export FM_USE_GAN=1
  export FM_LAMBDA_GAN=1.0
  export FM_GAN_SAMPLE_PROB=0.35
  export FM_GAN_SAMPLE_STEPS=12
  export FM_LAMBDA_SAMPLE_L1=0
  echo "phase1 joint_gan: FM + PatchGAN on ODE fakes (prob=${FM_GAN_SAMPLE_PROB})" >&2
}

# Phase 3: CFG v2 — learned null embedding, higher dropout, low test scale.
vanilla_fm_apply_joint_cfg_v2_env() {
  vanilla_fm_apply_joint_perc_pins
  export TRAIN_NAME=hemit_vanilla_fm_joint_cfg_v2
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
  unset FM_USE_FILM FM_USE_GAN
  export FM_USE_CFG=1
  export FM_NULL_MODE=learned
  export FM_CFG_DROPOUT=0.12
  export FM_CFG_SCALE=1.1
  echo "phase3 joint_cfg_v2: learned null dropout=${FM_CFG_DROPOUT} test_w=${FM_CFG_SCALE}" >&2
}

# Phase 4: FiLM v2 — per-marker head FiLM, identity reg, CD3-weighted L1.
vanilla_fm_apply_joint_film_v2_env() {
  vanilla_fm_apply_joint_perc_pins
  export TRAIN_NAME=hemit_vanilla_fm_joint_film_v2
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
  unset FM_USE_CFG FM_USE_GAN
  export FM_USE_FILM=1
  export FM_FILM_WHERE=head
  export FM_FILM_HIDDEN="${FM_FILM_HIDDEN:-128}"
  export FM_FILM_REG=0.01
  export FM_CHANNEL_WEIGHTS=1,1,1
  echo "phase4 joint_film_v2: head FiLM reg=${FM_FILM_REG} ch_weights=${FM_CHANNEL_WEIGHTS}" >&2
}

# Conditioning (finetune joint_perc/80): pseudo seg concat, bridge path, informed ODE init.
vanilla_fm_apply_joint_seg_env() {
  vanilla_fm_apply_joint_perc_pins
  export TRAIN_NAME=hemit_vanilla_fm_joint_seg
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
  export VANILLA_FM_COND_PROFILE=seg
  export DATASET_MODE=aligned_cond
  export FM_USE_SEG=1
  export FM_FLOW_PATH=noise
  export FM_INIT_FROM_COND=0
  unset FM_INIT_NOISE_SIGMA
  echo "joint_seg: H&E+seg concat, dataset=${DATASET_MODE}" >&2
}

vanilla_fm_apply_joint_bridge_env() {
  vanilla_fm_apply_joint_perc_pins
  export TRAIN_NAME=hemit_vanilla_fm_joint_bridge_v2
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
  export VANILLA_FM_COND_PROFILE=bridge
  export FM_USE_SEG=0
  unset DATASET_MODE
  export FM_FLOW_PATH=bridge
  export FM_INIT_FROM_COND=0
  unset FM_INIT_NOISE_SIGMA
  # v2: gray he_proj (no H&E green -> fake CD3), mix noise paths so CD3 is not forgotten
  export FM_HE_PROJ_INIT=gray
  export FM_BRIDGE_X0_SIGMA=0.05
  export FM_BRIDGE_NOISE_PROB=0.35
  echo "joint_bridge_v2: gray he_proj + bridge_noise_prob=${FM_BRIDGE_NOISE_PROB}" >&2
}

vanilla_fm_apply_joint_init_cond_env() {
  vanilla_fm_apply_joint_perc_pins
  export TRAIN_NAME=hemit_vanilla_fm_joint_init_cond
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
  export VANILLA_FM_COND_PROFILE=init_cond
  export FM_USE_SEG=0
  unset DATASET_MODE
  export FM_FLOW_PATH=noise
  export FM_INIT_FROM_COND=1
  export FM_INIT_NOISE_SIGMA="${FM_INIT_NOISE_SIGMA:-0.3}"
  echo "joint_init_cond: ODE start sigma*noise+(1-sigma)*proj(H&E), sigma=${FM_INIT_NOISE_SIGMA}" >&2
}

vanilla_fm_print_train_env() {
  echo "vanilla_fm config: locked=${VANILLA_FM_ENV_LOCKED:-0} TRAIN_NAME=${TRAIN_NAME:-?}"
  echo "  backbone=${FM_BACKBONE}  loss=${FM_LOSS}  channels=${FM_CHANNELS}  attn=${FM_ATTN}  res=${FM_RESBLOCKS}  up=${FM_UP_MODE:-bilinear}"
  if [[ "${FM_USE_CFG:-0}" == "1" ]]; then
    echo "  CFG: null=${FM_NULL_MODE:-zero} dropout=${FM_CFG_DROPOUT:-0.1} test_scale=${FM_CFG_SCALE:-1.5}"
  fi
  if [[ "${FM_USE_FILM:-0}" == "1" ]]; then
    echo "  FiLM: where=${FM_FILM_WHERE:-decoder} hidden=${FM_FILM_HIDDEN:-128} reg=${FM_FILM_REG:-0}"
  fi
  echo "  perc=${FM_LAMBDA_PERC}  time=${FM_TIME_DIST}  BATCH_SIZE=${BATCH_SIZE:-?}"
  echo "  FM_LAMBDA_L1=${FM_LAMBDA_L1}  FM_LAMBDA_SAMPLE_L1=${FM_LAMBDA_SAMPLE_L1}"
  echo "  infer: steps=${FM_STEPS}  ${FM_SAMPLE_METHOD}  val_steps=${FM_VAL_STEPS}"
  echo "  channel_L1_weights=${FM_CHANNEL_WEIGHTS:-1,2,1}"
  if [[ "${FM_USE_GAN:-0}" == "1" ]]; then
    echo "  GAN: lambda=${FM_LAMBDA_GAN:-1} prob=${FM_GAN_SAMPLE_PROB:-?} ode_steps=${FM_GAN_SAMPLE_STEPS:-?}"
  fi
  if [[ "${FM_USE_SEG:-0}" == "1" ]]; then
    echo "  cond: seg concat  dataset=${DATASET_MODE:-aligned_cond}"
  fi
  if [[ "${FM_FLOW_PATH:-noise}" != "noise" ]]; then
    echo "  flow_path=${FM_FLOW_PATH} he_proj_init=${FM_HE_PROJ_INIT:-gray} bridge_noise_prob=${FM_BRIDGE_NOISE_PROB:-0}"
  fi
  if [[ "${FM_INIT_FROM_COND:-0}" == "1" ]]; then
    echo "  init_from_cond: sigma=${FM_INIT_NOISE_SIGMA:-0.3}"
  fi
  if [[ -n "${VANILLA_FM_COND_PROFILE:-}" ]]; then
    echo "  cond_profile=${VANILLA_FM_COND_PROFILE} (verified when locked=1)"
  fi
  if [[ "${FM_BACKBONE}" == "monai" && -z "${FM_CROP_SIZE:-}" ]]; then
    echo "  WARNING: monai UNet at 1024² OOMs (mid-block attention). Use FM_BACKBONE=custom or FM_CROP_SIZE=512" >&2
  fi
}
