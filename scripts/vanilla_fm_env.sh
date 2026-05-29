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

vanilla_fm_print_train_env() {
  echo "vanilla_fm config:"
  echo "  backbone=${FM_BACKBONE}  loss=${FM_LOSS}  channels=${FM_CHANNELS}  attn=${FM_ATTN}  res=${FM_RESBLOCKS}"
  echo "  perc=${FM_LAMBDA_PERC}  time=${FM_TIME_DIST}  BATCH_SIZE=${BATCH_SIZE:-?}"
  echo "  FM_LAMBDA_L1=${FM_LAMBDA_L1}  FM_LAMBDA_SAMPLE_L1=${FM_LAMBDA_SAMPLE_L1}"
  echo "  infer: steps=${FM_STEPS}  ${FM_SAMPLE_METHOD}  val_steps=${FM_VAL_STEPS}"
  if [[ "${FM_BACKBONE}" == "monai" && -z "${FM_CROP_SIZE:-}" ]]; then
    echo "  WARNING: monai UNet at 1024² OOMs (mid-block attention). Use FM_BACKBONE=custom or FM_CROP_SIZE=512" >&2
  fi
}
