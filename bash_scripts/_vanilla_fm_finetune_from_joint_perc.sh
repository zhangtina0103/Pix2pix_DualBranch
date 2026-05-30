# shellcheck shell=bash
# Source after TRAIN_NAME is set. Seeds checkpoints/<TRAIN_NAME>/80_net_G.pth from joint_perc.
vanilla_fm_finetune_from_joint_perc_80() {
  local src="${REPO_ROOT}/checkpoints/hemit_vanilla_fm_joint_perc/80_net_G.pth"
  local dst="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/80_net_G.pth"
  [[ -f "${src}" ]] || {
    echo "ERROR: missing ${src} (train joint_perc to epoch 80 first)" >&2
    exit 1
  }
  mkdir -p "${REPO_ROOT}/checkpoints/${TRAIN_NAME}"
  cp -f "${src}" "${dst}"
  export CONTINUE_TRAIN=1
  export RESUME_FROM_EPOCH=80
  export EPOCH_COUNT="${EPOCH_COUNT:-81}"
  export N_EPOCHS="${N_EPOCHS:-70}"
  export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-30}"
}
