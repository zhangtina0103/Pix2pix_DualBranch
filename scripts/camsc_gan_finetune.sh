#!/usr/bin/env bash
# CaMSC GAN baselines: seed HEMIT checkpoint + finetune schedule (pix2pix / CUT / ASP).
# Source after camsc_scratch_env.sh.
#
#   export CAMSC_MODEL=pix2pix_ft   # or cut_ft | asp_ft
#   source scripts/camsc_gan_finetune.sh
#   camsc_gan_finetune_setup_from_hemit 0

camsc_gan_base_model() {
  case "${CAMSC_MODEL}" in
    pix2pix_ft) echo pix2pix ;;
    cut_ft) echo cut ;;
    asp_ft) echo asp ;;
    cyclegan_ft) echo cyclegan ;;
    *)
      echo "ERROR: CAMSC_MODEL=${CAMSC_MODEL} (need pix2pix_ft|cut_ft|asp_ft)" >&2
      return 1
      ;;
  esac
}

camsc_hemit_gan_ckpt_dir() {
  local base
  base="$(camsc_gan_base_model)"
  local name="${HEMIT_GAN_SRC_NAME:-}"
  if [[ -z "${name}" ]]; then
    case "${base}" in
      pix2pix) name="hemit_pix2pix_resnet9_512" ;;
      cut) name="hemit_cut_joint_512" ;;
      asp) name="hemit_asp_joint_512" ;;
      cyclegan) name="hemit_cyclegan_joint_512" ;;
    esac
  fi
  echo "${REPO_ROOT}/checkpoints/${name}"
}

camsc_seed_hemit_gan_ckpt() {
  local fold="${1:?fold index required}"
  local dest_name src_dir src_epoch dest_dir
  dest_name="$(camsc_resolved_train_name "${fold}")"
  src_epoch="${HEMIT_GAN_SRC_EPOCH:-80}"
  src_dir="$(camsc_hemit_gan_ckpt_dir)"
  dest_dir="${REPO_ROOT}/checkpoints/${dest_name}"
  [[ -d "${src_dir}" ]] || {
    echo "ERROR: missing HEMIT checkpoint dir ${src_dir}" >&2
    exit 1
  }
  shopt -s nullglob
  local files=( "${src_dir}/${src_epoch}_net_"*.pth )
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "ERROR: no ${src_epoch}_net_*.pth in ${src_dir}" >&2
    exit 1
  fi
  mkdir -p "${dest_dir}"
  local f base
  for f in "${files[@]}"; do
    base="$(basename "${f}")"
    cp -f "${f}" "${dest_dir}/${base}"
    echo "Seeded ${dest_dir}/${base} <- ${f}" >&2
  done
}

camsc_gan_finetune_apply_schedule() {
  local src_epoch="${HEMIT_GAN_SRC_EPOCH:-80}"
  local extra="${CAMSC_FT_EXTRA_EPOCHS:-30}"
  export CONTINUE_TRAIN=1
  export RESUME_FROM_EPOCH="${src_epoch}"
  export EPOCH_COUNT=$((src_epoch + 1))
  export N_EPOCHS=$((src_epoch + extra))
  export N_EPOCHS_DECAY="${CAMSC_FT_DECAY_EPOCHS:-0}"
  export TRAIN_LR="${TRAIN_LR:-5e-5}"
  export LR_DECAY_ITERS="${LR_DECAY_ITERS:-10}"
  export SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-5}"
  echo "CaMSC ${CAMSC_MODEL} finetune: resume@${src_epoch} epochs ${EPOCH_COUNT}..$((N_EPOCHS + N_EPOCHS_DECAY)) lr=${TRAIN_LR}" >&2
}

camsc_gan_finetune_setup_from_hemit() {
  local fold="${1:?fold index required}"
  camsc_seed_hemit_gan_ckpt "${fold}"
  camsc_gan_finetune_apply_schedule
}

camsc_gan_finetune_default_test_epoch() {
  local src_epoch="${HEMIT_GAN_SRC_EPOCH:-80}"
  local extra="${CAMSC_FT_EXTRA_EPOCHS:-30}"
  local decay="${CAMSC_FT_DECAY_EPOCHS:-0}"
  echo $((src_epoch + extra + decay))
}
