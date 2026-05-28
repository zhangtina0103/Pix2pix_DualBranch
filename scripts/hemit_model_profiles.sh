# Generator profiles — all use train.py / test.py / post_process.py (joint 3-channel).

MODEL="${MODEL:-dualbranch}"

case "${MODEL}" in
  pix2pix|resnet9)
    # Classic pix2pix baseline (this repo's train.py, not hemit/ per-marker pix2pix)
    NETG="${NETG:-resnet_9blocks}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_pix2pix_resnet9}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_pix2pix_resnet9}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    ;;
  dualbranch)
    NETG="${NETG:-SwinTResnet}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_SwinTResnet_New_2}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_SwinTResnet_New}"
    TRAIN_LR="${TRAIN_LR:-0.00003}"
    LAMBDA_L1="${LAMBDA_L1:-30}"
    ;;
  resnet6)
    NETG="${NETG:-resnet_6blocks}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_resnet6}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_resnet6}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    ;;
  unet256)
    NETG="${NETG:-unet_256}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_unet256}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_unet256}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    ;;
  unet128)
    NETG="${NETG:-unet_128}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_unet128}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_unet128}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    ;;
  unet1024)
    NETG="${NETG:-unet_1024}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_unet1024}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_unet1024}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    ;;
  swint)
    NETG="${NETG:-swinT}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_swinT}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_swinT}"
    TRAIN_LR="${TRAIN_LR:-0.00003}"
    LAMBDA_L1="${LAMBDA_L1:-30}"
    ;;
  swint_unet)
    NETG="${NETG:-SwinTUnet}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_SwinTUnet}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_SwinTUnet}"
    TRAIN_LR="${TRAIN_LR:-0.00003}"
    LAMBDA_L1="${LAMBDA_L1:-30}"
    ;;
  *)
    echo "Unknown MODEL=${MODEL}" >&2
    echo "  Native: dualbranch | pix2pix | resnet6 | unet256 | unet128 | unet1024 | swint | swint_unet" >&2
    echo "  Comparison (hemit/): cut | asp | cyclegan | fm | fm_plus" >&2
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
