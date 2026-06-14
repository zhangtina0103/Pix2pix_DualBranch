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

# Short finetune from full-data vanilla FM @130 (~15 ep ≈ 2–3h @ bs=4 512²).
# Usage: set TRAIN_NAME + env profile, then: vanilla_fm_finetune_from_joint_perc_full130 [extra_epochs]
vanilla_fm_finetune_from_joint_perc_full130() {
  local extra="${1:-15}"
  local src="${REPO_ROOT}/checkpoints/hemit_vanilla_fm_joint_perc_full/130_net_G.pth"
  local dst="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/130_net_G.pth"
  [[ -f "${src}" ]] || {
    echo "ERROR: missing ${src} (train joint_perc_full to epoch 130 first)" >&2
    exit 1
  }
  mkdir -p "${REPO_ROOT}/checkpoints/${TRAIN_NAME}"
  cp -f "${src}" "${dst}"
  export CONTINUE_TRAIN=1
  export RESUME_FROM_EPOCH=130
  export EPOCH_COUNT=131
  export N_EPOCHS=$((130 + extra))
  export N_EPOCHS_DECAY=0
  export TRAIN_LR="${TRAIN_LR:-5e-5}"
  echo "finetune from full/130: ${TRAIN_NAME} epochs ${EPOCH_COUNT}..$((N_EPOCHS + N_EPOCHS_DECAY)) (+${extra} ep) lr=${TRAIN_LR}" >&2
}

# Finetune from any checkpoint/<name>/<epoch>_net_G.pth (e.g. film_v2 @145).
vanilla_fm_finetune_from_checkpoint() {
  local src_name="$1" src_epoch="$2" extra_epochs="${3:-15}"
  local src="${REPO_ROOT}/checkpoints/${src_name}/${src_epoch}_net_G.pth"
  local dst="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${src_epoch}_net_G.pth"
  [[ -f "${src}" ]] || {
    echo "ERROR: missing ${src}" >&2
    exit 1
  }
  mkdir -p "${REPO_ROOT}/checkpoints/${TRAIN_NAME}"
  cp -f "${src}" "${dst}"
  export CONTINUE_TRAIN=1
  export RESUME_FROM_EPOCH="${src_epoch}"
  export EPOCH_COUNT=$((src_epoch + 1))
  export N_EPOCHS=$((src_epoch + extra_epochs))
  export N_EPOCHS_DECAY=0
  export TRAIN_LR="${TRAIN_LR:-5e-5}"
  echo "finetune from ${src_name}/${src_epoch}: ${TRAIN_NAME} epochs ${EPOCH_COUNT}..$((N_EPOCHS + N_EPOCHS_DECAY)) (+${extra_epochs} ep) lr=${TRAIN_LR}" >&2
}
