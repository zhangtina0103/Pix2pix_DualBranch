# Generator profiles — train.py → test.py → post_process.py (joint 3-channel multiplex).

MODEL="${MODEL:-dualbranch}"
# PY_MODEL is set per MODEL below (do not default to pix2pix before the case).

case "${MODEL}" in
  pix2pix|resnet9)
    PY_MODEL=pix2pix
    NETG="${NETG:-resnet_9blocks}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_pix2pix_resnet9}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_pix2pix_resnet9}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
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
    NETG="${NETG:-resnet_9blocks}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_cut_joint}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_cut_joint}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    LAMBDA_NCE="${LAMBDA_NCE:-1.0}"
    DATASET_MODE="${DATASET_MODE:-aligned}"
    ;;
  asp)
    PY_MODEL=asp
    NETG="${NETG:-resnet_9blocks}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_asp_joint}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_asp_joint}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    LAMBDA_ASP="${LAMBDA_ASP:-1.0}"
    DATASET_MODE="${DATASET_MODE:-aligned}"
    ;;
  cyclegan)
    PY_MODEL=cycle_gan
    NETG="${NETG:-resnet_9blocks}"
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
    # ~11M params with res=2 (verify: python scripts/count_fm_params.py)
    FM_CHANNELS="${FM_CHANNELS:-96,192,256}"
    FM_RESBLOCKS="${FM_RESBLOCKS:-2}"
    FM_STEPS="${FM_STEPS:-25}"
    VAL_FREQ="${VAL_FREQ:-10}"
    ;;
  *)
    echo "Unknown MODEL=${MODEL}" >&2
    echo "  Native: dualbranch | pix2pix | cut | asp | cyclegan | resnet6 | unet256 | ..." >&2
    echo "  FM (hemit/): fm | fm_plus" >&2
    echo "  Or set NETG and TRAIN_NAME yourself." >&2
    exit 1
    ;;
esac

N_EPOCHS="${N_EPOCHS:-50}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-30}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PRETRAINED_EPOCH="${PRETRAINED_EPOCH:-20}"
TEST_EPOCH="${TEST_EPOCH:-$((N_EPOCHS + N_EPOCHS_DECAY))}"
NUM_TEST="${NUM_TEST:-945}"
