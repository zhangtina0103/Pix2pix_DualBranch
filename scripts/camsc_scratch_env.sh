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
export CAMSC_KFOLDS="${CAMSC_KFOLDS:-5}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"

# CAMSC_MODEL: pix2pix | pix2pix_ft | cut | cut_ft | asp | asp_ft | cyclegan |
#              fm_cross_attn | fm_cross_attn_ft | fm_cross_attn_zeroshot | diffvs_zeroshot | vanilla_fm
export CAMSC_MODEL="${CAMSC_MODEL:-vanilla_fm}"

# Train-time aug only (random crops / flips / BF jitter in aligned_camsc loader):
#   export CAMSC_ENABLE_AUG=1
#   sbatch bash_scripts/train_camsc_bf_pix2pix_kfold.sbatch
camsc_apply_train_data_env() {
  if [[ "${CAMSC_ENABLE_AUG:-0}" != "1" ]]; then
    return 0
  fi
  export DATASET_MODE=aligned_camsc
  export PREPROCESS=crop
  export CROP_SIZE="${CROP_SIZE:-512}"
  export LOAD_SIZE="${LOAD_SIZE:-512}"
  export CAMSC_REPEATS="${CAMSC_REPEATS:-8}"
  export CAMSC_SCALE_JITTER="${CAMSC_SCALE_JITTER:-0.12}"
  export CAMSC_BF_NOISE="${CAMSC_BF_NOISE:-0.02}"
  echo "CaMSC train-time aug ON (dataset_mode=${DATASET_MODE}, repeats=${CAMSC_REPEATS})" >&2
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

# TRAIN_NAME after hemit_model_profiles.sh appends _${HEMIT_TRAIN_SIZE} when missing.
camsc_resolved_train_name() {
  local fold="${1:?fold}"
  local name
  name="$(camsc_fold_train_name "${fold}")"
  local size="${HEMIT_TRAIN_SIZE:-512}"
  if [[ "${size}" != "1024" ]] && [[ "${name}" != *"_${size}" ]]; then
    name="${name}_${size}"
  fi
  echo "${name}"
}

# Suffix after fold index for eval_camsc_kfold.py (e.g. _512_aug_512).
camsc_eval_name_suffix() {
  local fold="${1:-0}"
  local resolved prefix
  resolved="$(camsc_resolved_train_name "${fold}")"
  prefix="camsc_bf_${CAMSC_MODEL}_fold${fold}"
  echo "${resolved#"${prefix}"}"
}

camsc_results_name_prefix() {
  echo "camsc_bf_${CAMSC_MODEL}_fold"
}

_camsc_apply_fm_cross_attn_env() {
  export MODEL=vanilla_fm
  export PY_MODEL=vanilla_fm
  vanilla_fm_apply_fm_cross_attn_scratch_env
  unset FM_CROSS_ATTN_DECODER
  export FM_CROSS_ATTN_HEADS="${FM_CROSS_ATTN_HEADS:-4}"
  export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"
  export VANILLA_FM_ENV_LOCKED=1
  camsc_apply_train_data_env
}

# Apply model-specific training env. Call after sourcing vanilla_fm_env.sh (FM only).
camsc_apply_model_env() {
  case "${CAMSC_MODEL}" in
    pix2pix|pix2pix_ft)
      [[ "${CAMSC_MODEL}" == pix2pix_ft ]] && export CAMSC_MODEL=pix2pix_ft
      export MODEL=pix2pix
      export PY_MODEL=pix2pix
      export NETG="${NETG:-resnet_9blocks}"
      export NGF="${NGF:-64}"
      export LAMBDA_L1="${LAMBDA_L1:-100}"
      camsc_apply_train_data_env
      ;;
    cut|cut_ft)
      [[ "${CAMSC_MODEL}" == cut_ft ]] && export CAMSC_MODEL=cut_ft
      export MODEL=cut
      export PY_MODEL=cut
      export NETG="${NETG:-resnet_9blocks}"
      export NGF="${NGF:-64}"
      export LAMBDA_L1="${LAMBDA_L1:-100}"
      export LAMBDA_NCE="${LAMBDA_NCE:-1.0}"
      camsc_apply_train_data_env
      ;;
    asp|asp_ft)
      [[ "${CAMSC_MODEL}" == asp_ft ]] && export CAMSC_MODEL=asp_ft
      export MODEL=asp
      export PY_MODEL=asp
      export NETG="${NETG:-resnet_9blocks}"
      export NGF="${NGF:-64}"
      export LAMBDA_L1="${LAMBDA_L1:-100}"
      export LAMBDA_ASP="${LAMBDA_ASP:-1.0}"
      camsc_apply_train_data_env
      ;;
    cyclegan|cyclegan_ft)
      [[ "${CAMSC_MODEL}" == cyclegan_ft ]] && export CAMSC_MODEL=cyclegan_ft
      export MODEL=cyclegan
      export PY_MODEL=cycle_gan
      export NETG="${NETG:-resnet_9blocks}"
      export NGF="${NGF:-64}"
      export LAMBDA_A="${LAMBDA_A:-10.0}"
      export LAMBDA_B="${LAMBDA_B:-10.0}"
      export LAMBDA_IDENTITY="${LAMBDA_IDENTITY:-0.5}"
      camsc_apply_train_data_env
      ;;
    fm_cross_attn|cross_attn)
      export CAMSC_MODEL=fm_cross_attn
      _camsc_apply_fm_cross_attn_env
      ;;
    fm_cross_attn_ft|fm_cross_attn_finetune)
      export CAMSC_MODEL=fm_cross_attn_ft
      _camsc_apply_fm_cross_attn_env
      ;;
    fm_cross_attn_zeroshot|zeroshot)
      export CAMSC_MODEL=fm_cross_attn_zeroshot
      _camsc_apply_fm_cross_attn_env
      ;;
    vanilla_fm)
      export MODEL=vanilla_fm
      vanilla_fm_apply_train_env
      export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"
      ;;
    *)
      echo "ERROR: unknown CAMSC_MODEL=${CAMSC_MODEL}" >&2
      echo "  pix2pix | pix2pix_ft | cut | cut_ft | asp | asp_ft | cyclegan | cyclegan_ft |" >&2
      echo "  fm_cross_attn | fm_cross_attn_ft | fm_cross_attn_zeroshot | diffvs_zeroshot | vanilla_fm" >&2
      return 1
      ;;
  esac
}

camsc_run_fold_train() {
  local fold="${1:?fold}"
  camsc_apply_train_data_env
  export DATAROOT="$(camsc_fold_dataroot "${fold}")"
  export TRAIN_NAME="$(camsc_fold_train_name "${fold}")"
  export MODE=train
  case "${CAMSC_MODEL}" in
    pix2pix|cut|asp|cyclegan)
      camsc_apply_model_env
      echo "==> ${CAMSC_MODEL} fold=${fold} DATAROOT=${DATAROOT} TRAIN_NAME=${TRAIN_NAME}"
      bash scripts/run_hemit_native.sh
      ;;
    pix2pix_ft|cut_ft|asp_ft|cyclegan_ft)
      # shellcheck source=/dev/null
      source "${REPO_ROOT}/scripts/camsc_gan_finetune.sh"
      camsc_gan_finetune_setup_from_hemit "${fold}"
      camsc_apply_model_env
      echo "==> ${CAMSC_MODEL} fold=${fold} DATAROOT=${DATAROOT} TRAIN_NAME=${TRAIN_NAME}"
      bash scripts/run_hemit_native.sh
      ;;
    fm_cross_attn_ft)
      # shellcheck source=/dev/null
      source "${REPO_ROOT}/scripts/camsc_fm_finetune.sh"
      camsc_finetune_setup_from_hemit "${fold}"
      export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
      echo "==> fm_cross_attn_ft fold=${fold} DATAROOT=${DATAROOT} TRAIN_NAME=${TRAIN_NAME}"
      bash scripts/run_hemit_vanilla_fm.sh
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
    pix2pix|cut|asp|cyclegan|pix2pix_ft|cut_ft|asp_ft|cyclegan_ft)
      camsc_apply_model_env
      echo "==> ${CAMSC_MODEL} test fold=${fold} TRAIN_NAME=${TRAIN_NAME} epoch=${TEST_EPOCH:-?}"
      bash scripts/run_hemit_native.sh
      ;;
    *)
      export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
      bash scripts/run_hemit_vanilla_fm.sh
      ;;
  esac
}
