# shellcheck shell=bash
# Pin vanilla_fm training env. Source from run_hemit_vanilla_fm.sh / train sbatch.
#
# Standard FM (default): velocity MSE only — one UNet forward per step.
# Slow path: FM_USE_ODE_TRAIN=1  →  ODE sample-L1 in the train loop (OOM risk @ 1024²).

vanilla_fm_apply_train_env() {
  export FM_CHANNELS="${FM_CHANNELS:-96,192,272}"
  export FM_RESBLOCKS="${FM_RESBLOCKS:-2}"
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
  echo "  FM_CHANNELS=${FM_CHANNELS}  FM_RESBLOCKS=${FM_RESBLOCKS}  BATCH_SIZE=${BATCH_SIZE:-?}"
  echo "  train: FM_LAMBDA_L1=${FM_LAMBDA_L1}  FM_LAMBDA_SAMPLE_L1=${FM_LAMBDA_SAMPLE_L1}  (0/0 = standard FM)"
  echo "  infer: FM_STEPS=${FM_STEPS}  FM_SAMPLE_METHOD=${FM_SAMPLE_METHOD}  val_steps=${FM_VAL_STEPS}"
}
