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
  export EPOCH_COUNT=81
  # Always set (do not use ${N_EPOCHS:-70} — stale login/sbatch env can force 81→130).
  export N_EPOCHS=70
  export N_EPOCHS_DECAY=30
  echo "finetune schedule: epochs ${EPOCH_COUNT}..$((N_EPOCHS + N_EPOCHS_DECAY)) (short, from joint_perc/80)" >&2
}

# Longer low-LR finetune for cond_consistent (81 → 130).
# train.py: for epoch in range(epoch_count, n_epochs + n_epochs_decay + 1)
vanilla_fm_finetune_joint_opt_from_perc_80() {
  vanilla_fm_finetune_from_joint_perc_80
  export EPOCH_COUNT=81
  export N_EPOCHS=100
  export N_EPOCHS_DECAY=30
  echo "finetune schedule: epochs ${EPOCH_COUNT}..$((N_EPOCHS + N_EPOCHS_DECAY)) (long, cond_consistent)" >&2
}
