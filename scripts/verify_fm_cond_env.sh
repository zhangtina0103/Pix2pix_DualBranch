#!/usr/bin/env bash
# Dry-run: print + verify conditioning env (no GPU). Run on login node before sbatch.
#   bash scripts/verify_fm_cond_env.sh seg
#   bash scripts/verify_fm_cond_env.sh bridge
#   bash scripts/verify_fm_cond_env.sh init_cond
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export REPO_ROOT="${ROOT}"
PROFILE="${1:?usage: verify_fm_cond_env.sh seg|bridge|init_cond|init_only|seg_only|consistent|consistent_scratch|consistent_v2|cellpose|patchnce|res3}"

# shellcheck source=/dev/null
source "${ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
# shellcheck source=/dev/null
source "${ROOT}/scripts/vanilla_fm_env.sh"

case "${PROFILE}" in
  seg) vanilla_fm_apply_joint_seg_env ;;
  bridge) vanilla_fm_apply_joint_bridge_env ;;
  init_cond) vanilla_fm_apply_joint_init_cond_env ;;
  init_only) vanilla_fm_apply_cond_init_only_env ;;
  seg_only) vanilla_fm_apply_cond_seg_only_env ;;
  consistent|opt) vanilla_fm_apply_cond_consistent_env ;;
  consistent_scratch) vanilla_fm_apply_cond_consistent_scratch_env ;;
  consistent_v2) vanilla_fm_apply_cond_consistent_v2_env ;;
  cellpose|mentor_cellpose) vanilla_fm_apply_joint_perc_cellpose_env ;;
  patchnce|mentor_patchnce) vanilla_fm_apply_joint_perc_patchnce_env ;;
  res3|mentor_res3) vanilla_fm_apply_joint_perc_res3_env ;;
  monai512) vanilla_fm_apply_monai512_env ;;
  *) echo "unknown profile: ${PROFILE}" >&2; exit 1 ;;
esac

export VANILLA_FM_ENV_LOCKED=1
export MODEL=vanilla_fm
# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_model_profiles.sh"
vanilla_fm_verify_locked_env
vanilla_fm_print_train_env
echo "OK: ${PROFILE} env verified (no overrides detected)"
