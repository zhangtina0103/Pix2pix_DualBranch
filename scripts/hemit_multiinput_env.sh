# shellcheck shell=bash
# HEMIT multi-input CD3/panCK experiments (FM + cross-attention @ 80 epochs).
#
#   sbatch bash_scripts/prepare_hemit_multiinput.sbatch
#   HEMIT_MULTI_VARIANT=he sbatch bash_scripts/train_hemit_multiinput.sbatch
#   HEMIT_MULTI_VARIANT=he_dapi sbatch bash_scripts/eval_hemit_multiinput.sbatch

hemit_multiinput_apply_env() {
  local variant="${HEMIT_MULTI_VARIANT:?Set HEMIT_MULTI_VARIANT=he|dapi|he_dapi}"
  local _root
  _root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

  # shellcheck source=/dev/null
  source "${_root}/scripts/hemit_scratch_env.sh"
  hemit_scratch_ensure_dirs

  # shellcheck source=/dev/null
  source "${_root}/scripts/vanilla_fm_env.sh"
  vanilla_fm_clear_stale_env
  vanilla_fm_apply_joint_perc_pins
  vanilla_fm_apply_fm_scratch_80_schedule

  export INPUT_NC=3
  export OUTPUT_NC=3
  # [CD3, panCK, pad] — match fair main HEMIT (1,1,1 on DAPI/CD3/panCK), zero pad
  export FM_CHANNEL_WEIGHTS=1,1,0
  export FM_USE_TRI_HEAD=0
  export FM_USE_CROSS_ATTN=1
  export FM_CROSS_ATTN_DECODER="${FM_CROSS_ATTN_DECODER:-0}"
  export FM_CROSS_ATTN_HEADS="${FM_CROSS_ATTN_HEADS:-4}"
  export FM_FLOW_PATH=noise
  export FM_LAMBDA_SAMPLE_L1=0
  unset FM_INIT_FROM_COND FM_HE_PROJ_INIT FM_USE_PATCHNCE

  export HEMIT_METRICS_SCRIPT="${_root}/scripts/eval_hemit_multiinput.py"
  export HEMIT_TRAIN_SIZE="${HEMIT_TRAIN_SIZE:-512}"
  export DISPLAY_ID="${DISPLAY_ID:--1}"

  case "${variant}" in
    he)
      export TRAIN_NAME=hemit_multi_he_cd3panck
      export DATAROOT="${DATAROOT:-$(hemit_multiinput_dataroot he)}"
      unset DATASET_MODE FM_USE_SEG
      ;;
    dapi)
      export TRAIN_NAME=hemit_multi_dapi_cd3panck
      export DATAROOT="${DATAROOT:-$(hemit_multiinput_dataroot dapi)}"
      unset DATASET_MODE FM_USE_SEG
      ;;
    he_dapi)
      export TRAIN_NAME=hemit_multi_he_dapi_cd3panck
      export DATAROOT="${DATAROOT:-$(hemit_multiinput_dataroot he_dapi)}"
      export DATASET_MODE=aligned_cond
      export FM_USE_SEG=1
      ;;
    *)
      echo "ERROR: HEMIT_MULTI_VARIANT must be he, dapi, or he_dapi (got ${variant})" >&2
      return 1
      ;;
  esac

  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
  export VANILLA_FM_ENV_LOCKED=1
  echo "hemit_multiinput: variant=${variant} TRAIN_NAME=${TRAIN_NAME} DATAROOT=${DATAROOT}" >&2
  echo "  dataset_mode=${DATASET_MODE:-aligned} fm_use_seg=${FM_USE_SEG:-0} ch_weights=${FM_CHANNEL_WEIGHTS}" >&2
}

hemit_multiinput_datadir() {
  hemit_multiinput_dataroot "${HEMIT_MULTI_VARIANT:?}"
}
