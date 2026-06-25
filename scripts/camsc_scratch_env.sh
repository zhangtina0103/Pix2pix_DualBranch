# CaMSC brightfield paths on ORCD Engaging (override before sourcing if needed).
#
#   source scripts/camsc_scratch_env.sh
#
# Raw TIFs:  ~/orcd/scratch/camsc/20260504/
# K-fold:     ~/orcd/scratch/camsc/datasets/camsc_bf_kfold/fold{0..4}/

if [[ -z "${CAMSC_SCRATCH_ROOT:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    CAMSC_SCRATCH_ROOT="${SCRATCH}/camsc"
  else
    CAMSC_SCRATCH_ROOT="${HOME}/orcd/scratch/camsc"
  fi
fi

export CAMSC_SCRATCH_ROOT
export CAMSC_SRC="${CAMSC_SRC:-${CAMSC_SCRATCH_ROOT}/20260504}"
export CAMSC_KFOLD_ROOT="${CAMSC_KFOLD_ROOT:-${CAMSC_SCRATCH_ROOT}/datasets/camsc_bf_kfold}"
export CAMSC_KFOLD_ROOT_AUG="${CAMSC_KFOLD_ROOT_AUG:-${CAMSC_SCRATCH_ROOT}/datasets/camsc_bf_kfold_aug}"
export CAMSC_KFOLDS="${CAMSC_KFOLDS:-5}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"

# CAMSC_MODEL: pix2pix | fm_cross_attn | vanilla_fm
export CAMSC_MODEL="${CAMSC_MODEL:-vanilla_fm}"

# Retrain with full-res + random 512 crops, flips, rot90, BF jitter:
#   export CAMSC_ENABLE_AUG=1
#   sbatch bash_scripts/prepare_camsc_bf.sbatch   # writes camsc_bf_kfold_aug
camsc_apply_train_data_env() {
  if [[ "${CAMSC_ENABLE_AUG:-0}" != "1" ]]; then
    return 0
  fi
  export CAMSC_KFOLD_ROOT="${CAMSC_KFOLD_ROOT_AUG}"
  export DATASET_MODE=aligned_camsc
  export PREPROCESS=crop
  export CROP_SIZE="${CROP_SIZE:-512}"
  export LOAD_SIZE="${LOAD_SIZE:-512}"
  export CAMSC_REPEATS="${CAMSC_REPEATS:-8}"
  export CAMSC_SCALE_JITTER="${CAMSC_SCALE_JITTER:-0.12}"
  export CAMSC_BF_NOISE="${CAMSC_BF_NOISE:-0.02}"
  echo "CaMSC aug ON → ${CAMSC_KFOLD_ROOT} (full-res, random ${CROP_SIZE}px crops, repeats=${CAMSC_REPEATS})" >&2
}

camsc_fold_dataroot() {
  local fold="${1:?fold index required}"
  echo "${CAMSC_KFOLD_ROOT}/fold${fold}"
}

camsc_fold_train_name() {
  local fold="${1:?fold index required}"
  local model="${CAMSC_MODEL:-vanilla_fm}"
  local size="${HEMIT_TRAIN_SIZE:-512}"
  local aug_suffix=""
  [[ "${CAMSC_ENABLE_AUG:-0}" == "1" ]] && aug_suffix="_aug"
  echo "camsc_bf_${model}_fold${fold}_${size}${aug_suffix}"
}

camsc_results_name_prefix() {
  echo "camsc_bf_${CAMSC_MODEL}_fold"
}

# Apply model-specific training env. Call after sourcing vanilla_fm_env.sh (FM only).
camsc_apply_model_env() {
  case "${CAMSC_MODEL}" in
    pix2pix)
      export MODEL=pix2pix
      export PY_MODEL=pix2pix
      export NETG="${NETG:-resnet_9blocks}"
      export NGF="${NGF:-64}"
      export LAMBDA_L1="${LAMBDA_L1:-100}"
      ;;
    fm_cross_attn|cross_attn)
      export CAMSC_MODEL=fm_cross_attn
      export MODEL=vanilla_fm
      export PY_MODEL=vanilla_fm
      vanilla_fm_apply_fm_cross_attn_scratch_env
      unset FM_CROSS_ATTN_DECODER
      export FM_CROSS_ATTN_HEADS="${FM_CROSS_ATTN_HEADS:-4}"
      export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"
      export VANILLA_FM_ENV_LOCKED=1
      ;;
    vanilla_fm)
      export MODEL=vanilla_fm
      vanilla_fm_apply_train_env
      export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"
      ;;
    *)
      echo "ERROR: unknown CAMSC_MODEL=${CAMSC_MODEL} (pix2pix | fm_cross_attn | vanilla_fm)" >&2
      return 1
      ;;
  esac
}

camsc_run_fold_train() {
  local fold="${1:?fold}"
  export DATAROOT="$(camsc_fold_dataroot "${fold}")"
  export TRAIN_NAME="$(camsc_fold_train_name "${fold}")"
  export MODE=train
  case "${CAMSC_MODEL}" in
    pix2pix)
      export MODEL=pix2pix
      export PY_MODEL=pix2pix
      export NETG="${NETG:-resnet_9blocks}"
      export NGF="${NGF:-64}"
      export LAMBDA_L1="${LAMBDA_L1:-100}"
      echo "==> pix2pix fold=${fold} DATAROOT=${DATAROOT} TRAIN_NAME=${TRAIN_NAME}"
      bash scripts/run_hemit_native.sh
      ;;
    *)
      export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
      bash scripts/run_hemit_vanilla_fm.sh
      ;;
  esac
}

camsc_run_fold_test() {
  local fold="${1:?fold}"
  export DATAROOT="$(camsc_fold_dataroot "${fold}")"
  export TRAIN_NAME="$(camsc_fold_train_name "${fold}")"
  export MODE=test
  case "${CAMSC_MODEL}" in
    pix2pix)
      export MODEL=pix2pix
      export PY_MODEL=pix2pix
      export NETG="${NETG:-resnet_9blocks}"
      export NGF="${NGF:-64}"
      echo "==> pix2pix test fold=${fold} TRAIN_NAME=${TRAIN_NAME} epoch=${TEST_EPOCH:-?}"
      bash scripts/run_hemit_native.sh
      ;;
    *)
      export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
      bash scripts/run_hemit_vanilla_fm.sh
      ;;
  esac
}
