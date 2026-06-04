# Generator profiles — train.py → test.py → post_process.py (joint 3-channel multiplex).

MODEL="${MODEL:-dualbranch}"
# PY_MODEL is set per MODEL below (do not default to pix2pix before the case).

case "${MODEL}" in
  pix2pix|resnet9)
    PY_MODEL=pix2pix
    # ResNet9 G ~11.38M @ ngf=64 (scripts/count_hemit_g_params.py)
    NETG=resnet_9blocks
    NGF="${NGF:-64}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_pix2pix_resnet9}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_pix2pix_resnet9}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    ;;
  pix2pixhd)
    # U-Net @ 1024². ngf=64 is ~50–80M G — NOT param-matched. Default ngf=24 targets ~11M (verify on cluster).
    # Fair ResNet baseline = MODEL=pix2pix (resnet_9blocks ngf=64).
    PY_MODEL=pix2pix
    NETG="${NETG:-unet_1024}"
    NGF="${NGF:-24}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_pix2pixhd_unet1024_ngf24}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_pix2pixhd_unet1024_ngf24}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    ;;
  dualbranch)
    PY_MODEL=pix2pix
    NETG="${NETG:-SwinTResnet}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_SwinTResnet_New_2}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_SwinTResnet_New}"
    TRAIN_LR="${TRAIN_LR:-0.00003}"
    LAMBDA_L1="${LAMBDA_L1:-30}"
    ;;
  resnet6)
    PY_MODEL=pix2pix
    NETG="${NETG:-resnet_6blocks}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_resnet6}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_resnet6}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    ;;
  unet256)
    PY_MODEL=pix2pix
    NETG="${NETG:-unet_256}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_unet256}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_unet256}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    ;;
  unet128)
    PY_MODEL=pix2pix
    NETG="${NETG:-unet_128}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_unet128}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_unet128}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    ;;
  unet1024)
    PY_MODEL=pix2pix
    NETG="${NETG:-unet_1024}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_unet1024}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_unet1024}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    ;;
  swint)
    PY_MODEL=pix2pix
    NETG="${NETG:-swinT}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_swinT}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_swinT}"
    TRAIN_LR="${TRAIN_LR:-0.00003}"
    LAMBDA_L1="${LAMBDA_L1:-30}"
    ;;
  swint_unet)
    PY_MODEL=pix2pix
    NETG="${NETG:-SwinTUnet}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_SwinTUnet}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_SwinTUnet}"
    TRAIN_LR="${TRAIN_LR:-0.00003}"
    LAMBDA_L1="${LAMBDA_L1:-30}"
    ;;
  cut)
    PY_MODEL=cut
    NETG=resnet_9blocks
    NGF="${NGF:-64}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_cut_joint}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_cut_joint}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    LAMBDA_NCE="${LAMBDA_NCE:-1.0}"
    DATASET_MODE="${DATASET_MODE:-aligned}"
    ;;
  asp)
    PY_MODEL=asp
    NETG=resnet_9blocks
    NGF="${NGF:-64}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_asp_joint}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_asp_joint}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    LAMBDA_ASP="${LAMBDA_ASP:-1.0}"
    DATASET_MODE="${DATASET_MODE:-aligned}"
    ;;
  cyclegan)
    PY_MODEL=cycle_gan
    NETG=resnet_9blocks
    NGF="${NGF:-64}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_cyclegan_joint}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_cyclegan_joint}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_A="${LAMBDA_A:-10.0}"
    LAMBDA_B="${LAMBDA_B:-10.0}"
    LAMBDA_IDENTITY="${LAMBDA_IDENTITY:-0.5}"
    DATASET_MODE="${DATASET_MODE:-aligned}"
    ;;
  vanilla_fm)
    PY_MODEL=vanilla_fm
    TRAIN_NAME="${TRAIN_NAME:-hemit_vanilla_fm_joint}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_vanilla_fm_joint}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    DATASET_MODE="${DATASET_MODE:-aligned}"
    DISPLAY_ID="${DISPLAY_ID:--1}"
    VAL_FREQ="${VAL_FREQ:-10}"
    # When VANILLA_FM_ENV_LOCKED=1, FM_* come only from vanilla_fm_apply_*_env (not profile defaults).
    if [[ "${VANILLA_FM_ENV_LOCKED:-0}" != "1" ]]; then
      FM_BACKBONE="${FM_BACKBONE:-custom}"
      FM_LOSS="${FM_LOSS:-x1}"
      FM_CHANNELS="${FM_CHANNELS:-64,128,192}"
      FM_ATTN="${FM_ATTN:-0,0,0}"
      FM_RESBLOCKS="${FM_RESBLOCKS:-2}"
      FM_LAMBDA_PERC="${FM_LAMBDA_PERC:-0.1}"
      FM_STEPS="${FM_STEPS:-25}"
      FM_VAL_STEPS="${FM_VAL_STEPS:-8}"
      FM_SAMPLE_METHOD="${FM_SAMPLE_METHOD:-heun}"
      FM_SAMPLE_L1_PROB="${FM_SAMPLE_L1_PROB:-1.0}"
      FM_TRAIN_SAMPLE_METHOD="${FM_TRAIN_SAMPLE_METHOD:-euler}"
    fi
    ;;
  *)
    echo "Unknown MODEL=${MODEL}" >&2
    echo "  Native: dualbranch | pix2pix | pix2pixhd | cut | asp | cyclegan | resnet6 | unet256 | ..." >&2
    echo "  FM (hemit/): fm | fm_plus" >&2
    echo "  Or set NETG and TRAIN_NAME yourself." >&2
    exit 1
    ;;
esac

N_EPOCHS="${N_EPOCHS:-50}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-30}"

# Fair HEMIT sweep default: 512² (~930 steps/epoch @ bs=4). Full res: HEMIT_TRAIN_SIZE=1024
HEMIT_TRAIN_SIZE="${HEMIT_TRAIN_SIZE:-512}"
LOAD_SIZE="${LOAD_SIZE:-${HEMIT_TRAIN_SIZE}}"
CROP_SIZE="${CROP_SIZE:-${HEMIT_TRAIN_SIZE}}"
if [[ "${HEMIT_TRAIN_SIZE}" != "1024" ]]; then
  if [[ "${TRAIN_NAME}" != *"_${HEMIT_TRAIN_SIZE}" ]]; then
    TRAIN_NAME="${TRAIN_NAME}_${HEMIT_TRAIN_SIZE}"
    PRETRAINED_NAME="${PRETRAINED_NAME}_${HEMIT_TRAIN_SIZE}"
  fi
fi
export HEMIT_TRAIN_SIZE LOAD_SIZE CROP_SIZE

# vanilla_fm_apply_* sets VANILLA_FM_EXPECTED_TRAIN_NAME before this suffix runs
if [[ "${MODEL}" == "vanilla_fm" && "${VANILLA_FM_ENV_LOCKED:-0}" == "1" ]]; then
  export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
fi

# Per-model batch defaults (only if BATCH_SIZE unset on sbatch cmdline).
case "${MODEL}" in
  pix2pix|resnet9) BATCH_SIZE="${BATCH_SIZE:-4}" ;;
  pix2pixhd) BATCH_SIZE="${BATCH_SIZE:-1}" ;;
  cut|asp)
    if [[ "${CROP_SIZE}" -ge 1024 ]]; then
      BATCH_SIZE="${BATCH_SIZE:-1}"
    else
      BATCH_SIZE="${BATCH_SIZE:-2}"   # bs=4 + PatchNCE OOMs on L40S even @ 512²
    fi
    ;;
  cyclegan) BATCH_SIZE="${BATCH_SIZE:-4}" ;;
  vanilla_fm)
    if [[ "${CROP_SIZE}" -ge 1024 ]]; then
      BATCH_SIZE="${BATCH_SIZE:-1}"
    else
      BATCH_SIZE="${BATCH_SIZE:-4}"   # 512² ~930 steps/ep; OOM → BATCH_SIZE=2
    fi
    ;;
  *) BATCH_SIZE="${BATCH_SIZE:-2}" ;;
esac
PRETRAINED_EPOCH="${PRETRAINED_EPOCH:-20}"
TEST_EPOCH="${TEST_EPOCH:-$((N_EPOCHS + N_EPOCHS_DECAY))}"
NUM_TEST="${NUM_TEST:-945}"
