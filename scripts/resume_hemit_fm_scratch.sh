#!/usr/bin/env bash
# Resume a scratch FM run (6h walltime chunks). Same TRAIN_NAME + env profile as train job.
#
#   export RESUME_PROFILE=advanced   # or: v2, consistent_scratch, tri_head, cross_attn, ...
#   export RESUME_FROM_EPOCH=25      # optional; default = latest saved *_net_G.pth
#   bash scripts/resume_hemit_fm_scratch.sh
#
# Cluster:
#   RESUME_PROFILE=advanced sbatch bash_scripts/resume_hemit_fm_scratch.sbatch
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MODEL=vanilla_fm
export MODE=train
export CONTINUE_TRAIN=1

export RESUME_PROFILE="${RESUME_PROFILE:?Set RESUME_PROFILE (joint_perc|joint_perc_scratch|advanced|...)}"

# shellcheck source=/dev/null
source "${ROOT}/scripts/vanilla_fm_env.sh"

case "${RESUME_PROFILE}" in
  advanced) vanilla_fm_apply_cond_fm_advanced_scratch_env ;;
  advanced_ode) vanilla_fm_apply_cond_fm_advanced_ode_scratch_env ;;
  v2|consistent_v2) vanilla_fm_apply_cond_consistent_v2_env ;;
  consistent_scratch) vanilla_fm_apply_cond_consistent_scratch_env ;;
  light) vanilla_fm_apply_cond_light_scratch_env ;;
  seg_only|seg_only_scratch) vanilla_fm_apply_cond_seg_only_scratch_env ;;
  bridge_scratch) vanilla_fm_apply_cond_bridge_scratch_env ;;
  seg_init|seg_init_scratch) vanilla_fm_apply_cond_seg_init_scratch_env ;;
  tri_head) vanilla_fm_apply_fm_tri_head_scratch_env ;;
  cross_attn) vanilla_fm_apply_fm_cross_attn_scratch_env ;;
  perc_strong) vanilla_fm_apply_perc_strong_scratch_env ;;
  res3) vanilla_fm_apply_joint_perc_res3_scratch_env ;;
  monai512) vanilla_fm_apply_monai512_env ;;
  joint_perc) vanilla_fm_apply_joint_perc_env ;;
  joint_perc_scratch) vanilla_fm_apply_joint_perc_scratch_env ;;
  joint_perc_full) vanilla_fm_apply_joint_perc_fulldata_env ;;
  consistent) vanilla_fm_apply_cond_consistent_env ;;
  *) echo "ERROR: unknown RESUME_PROFILE=${RESUME_PROFILE}" >&2; exit 1 ;;
esac

export VANILLA_FM_ENV_LOCKED=1
export CONTINUE_TRAIN=1

# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_model_profiles.sh"
export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
vanilla_fm_verify_locked_env

ckpt_dir="${ROOT}/checkpoints/${TRAIN_NAME}"
if [[ ! -d "${ckpt_dir}" ]]; then
  echo "ERROR: missing ${ckpt_dir}" >&2
  exit 1
fi

if [[ -z "${RESUME_FROM_EPOCH:-}" ]]; then
  RESUME_FROM_EPOCH="$(
    find "${ckpt_dir}" -maxdepth 1 -name '*_net_G.pth' ! -name 'latest_net_G.pth' -printf '%f\n' 2>/dev/null \
      | sed -n 's/^\([0-9][0-9]*\)_net_G\.pth$/\1/p' | sort -n | tail -1
  )"
  if [[ -z "${RESUME_FROM_EPOCH}" ]] && [[ -f "${ckpt_dir}/latest_net_G.pth" ]]; then
    echo "WARN: only latest_net_G.pth found; set RESUME_FROM_EPOCH explicitly" >&2
    exit 1
  fi
fi

export RESUME_FROM_EPOCH
export CONTINUE_TRAIN=1
export EPOCH_COUNT="${EPOCH_COUNT:-$((RESUME_FROM_EPOCH + 1))}"
_end=$((N_EPOCHS + N_EPOCHS_DECAY))

if (( EPOCH_COUNT > _end )); then
  echo "Training complete: ${TRAIN_NAME} @ epoch ${_end}"
  exit 0
fi

if (( RESUME_FROM_EPOCH >= N_EPOCHS )); then
  export TRAIN_LR="${TRAIN_LR:-0.00002}"
else
  export TRAIN_LR="${TRAIN_LR:-0.0002}"
fi

vanilla_fm_print_train_env
echo "===== resume FM: ${TRAIN_NAME} load@${RESUME_FROM_EPOCH} train epochs ${EPOCH_COUNT}..${_end} (N_EPOCHS=${N_EPOCHS}+decay=${N_EPOCHS_DECAY}) ====="

exec bash "${ROOT}/scripts/run_hemit_vanilla_fm.sh"
