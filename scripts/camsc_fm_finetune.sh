#!/usr/bin/env bash
# CaMSC FM cross-attn: seed HEMIT checkpoint + finetune schedule.
# Source after camsc_scratch_env.sh (needs REPO_ROOT, camsc_resolved_train_name).
#
#   source scripts/camsc_fm_finetune.sh
#   camsc_finetune_setup_from_hemit 0

camsc_hemit_cross_attn_ckpt_path() {
  local epoch="${1:-${HEMIT_CROSS_ATTN_SRC_EPOCH:-80}}"
  local name="${HEMIT_CROSS_ATTN_SRC_NAME:-hemit_fm_cross_attn_scratch_512}"
  echo "${REPO_ROOT}/checkpoints/${name}/${epoch}_net_G.pth"
}

camsc_seed_hemit_cross_attn_ckpt() {
  local fold="${1:?fold index required}"
  local dest_name src_epoch src dest
  dest_name="$(camsc_resolved_train_name "${fold}")"
  src_epoch="${HEMIT_CROSS_ATTN_SRC_EPOCH:-80}"
  src="$(camsc_hemit_cross_attn_ckpt_path "${src_epoch}")"
  dest="${REPO_ROOT}/checkpoints/${dest_name}/${src_epoch}_net_G.pth"
  [[ -f "${src}" ]] || {
    echo "ERROR: missing HEMIT checkpoint ${src}" >&2
    echo "  Train hemit_fm_cross_attn_scratch to epoch ${src_epoch}, or set HEMIT_CROSS_ATTN_SRC_NAME." >&2
    exit 1
  }
  mkdir -p "${REPO_ROOT}/checkpoints/${dest_name}"
  cp -f "${src}" "${dest}"
  echo "Seeded ${dest} <- ${src}" >&2
}

camsc_finetune_apply_schedule() {
  local src_epoch="${HEMIT_CROSS_ATTN_SRC_EPOCH:-80}"
  local extra="${CAMSC_FT_EXTRA_EPOCHS:-30}"
  export CONTINUE_TRAIN=1
  export RESUME_FROM_EPOCH="${src_epoch}"
  export EPOCH_COUNT=$((src_epoch + 1))
  export N_EPOCHS=$((src_epoch + extra))
  export N_EPOCHS_DECAY="${CAMSC_FT_DECAY_EPOCHS:-0}"
  export TRAIN_LR="${TRAIN_LR:-5e-5}"
  export LR_DECAY_ITERS="${LR_DECAY_ITERS:-10}"
  export SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
  echo "CaMSC finetune: resume@${src_epoch} epochs ${EPOCH_COUNT}..$((N_EPOCHS + N_EPOCHS_DECAY)) lr=${TRAIN_LR}" >&2
}

camsc_finetune_setup_from_hemit() {
  local fold="${1:?fold index required}"
  camsc_seed_hemit_cross_attn_ckpt "${fold}"
  camsc_finetune_apply_schedule
}

camsc_finetune_default_test_epoch() {
  local src_epoch="${HEMIT_CROSS_ATTN_SRC_EPOCH:-80}"
  local extra="${CAMSC_FT_EXTRA_EPOCHS:-30}"
  local decay="${CAMSC_FT_DECAY_EPOCHS:-0}"
  echo $((src_epoch + extra + decay))
}
