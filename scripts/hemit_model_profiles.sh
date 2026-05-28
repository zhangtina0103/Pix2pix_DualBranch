# Sourced by run_hemit_all.sh — sets variables from MODEL.
# Native models use train.py / test.py / post_process.py (joint 3-channel multiplex).

case "${MODEL}" in
  dualbranch)
    NETG="${NETG:-SwinTResnet}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_SwinTResnet_New_2}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_SwinTResnet_New}"
    N_EPOCHS="${N_EPOCHS:-50}"
    N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-30}"
    TRAIN_LR="${TRAIN_LR:-0.00003}"
    LAMBDA_L1="${LAMBDA_L1:-30}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
  ;;
  pix2pix_resnet9)
    NETG="${NETG:-resnet_9blocks}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_resnet9}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_resnet9}"
    N_EPOCHS="${N_EPOCHS:-50}"
    N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-30}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
  ;;
  pix2pix_unet256)
    NETG="${NETG:-unet_256}"
    TRAIN_NAME="${TRAIN_NAME:-hemit_unet256}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-hemit_unet256}"
    N_EPOCHS="${N_EPOCHS:-50}"
    N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-30}"
    TRAIN_LR="${TRAIN_LR:-0.0002}"
    LAMBDA_L1="${LAMBDA_L1:-100}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
  ;;
  vs_*)
    : # handled in hemit_comparison/
  ;;
  *)
    echo "Unknown MODEL=${MODEL}" >&2
    return 1
  ;;
esac

PRETRAINED_EPOCH="${PRETRAINED_EPOCH:-20}"
TEST_EPOCH="${TEST_EPOCH:-$((N_EPOCHS + N_EPOCHS_DECAY))}"
NUM_TEST="${NUM_TEST:-945}"
