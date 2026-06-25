#!/usr/bin/env bash
# Submit eval @ TEST_EPOCH (default 80) for every ablation run that has a checkpoint.
# Does NOT train. Run audit first:
#   bash scripts/audit_hemit_ablation_epoch80.sh
# Then:
#   bash scripts/batch_eval_hemit_ablation_epoch80.sh
#   bash scripts/batch_eval_hemit_ablation_epoch80.sh --dry-run
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EPOCH="${TEST_EPOCH:-80}"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

# TRAIN_NAME -> eval sbatch (must set vanilla_fm env inside sbatch)
declare -A EVAL=(
  [hemit_vanilla_fm_joint_perc]=eval_hemit_vanilla_fm_joint_perc.sbatch
  [hemit_vanilla_fm_joint_perc_full]=eval_hemit_vanilla_fm_joint_perc_full.sbatch
  [hemit_fm_cross_attn_scratch]=eval_hemit_fm_cross_attn_scratch.sbatch
  [hemit_cond_fm_seg_only_scratch]=eval_hemit_cond_fm_seg_only_scratch.sbatch
  [hemit_cond_fm_seg_init_scratch]=eval_hemit_cond_fm_seg_init_scratch.sbatch
  [hemit_cond_fm_bridge_scratch]=eval_hemit_cond_fm_bridge_scratch.sbatch
  [hemit_fm_cross_attn_patchnce_scratch]=eval_hemit_fm_cross_attn_patchnce_scratch.sbatch
  [hemit_fm_cross_attn_init_scratch]=eval_hemit_fm_cross_attn_init_scratch.sbatch
  [hemit_fm_cross_attn_scratch_focal]=eval_hemit_fm_cross_attn_scratch_focal.sbatch
  [hemit_fm_cross_attn_scratch_focal_tuned]=eval_hemit_fm_cross_attn_scratch_focal_tuned.sbatch
  [hemit_fm_cross_attn_scratch_focal_g075]=eval_hemit_fm_cross_attn_scratch_focal_g075.sbatch
  [hemit_fm_cross_attn_scratch_focal_tuned_vel]=eval_hemit_fm_cross_attn_scratch_focal_tuned_vel.sbatch
  [hemit_fm_cross_attn_scratch_focal_cd3]=eval_hemit_fm_cross_attn_scratch_focal_cd3.sbatch
  [hemit_fm_cross_attn_scratch_cd3]=eval_hemit_fm_cross_attn_scratch_cd3.sbatch
  [hemit_fm_cross_attn_scratch_fg]=eval_hemit_fm_cross_attn_scratch_fg.sbatch
  [hemit_fm_cross_attn_scratch_vel]=eval_hemit_fm_cross_attn_scratch_vel.sbatch
)

# Finetune variants: test_hemit_* sbatch (subset protocol — use only if no eval_* exists)
declare -A TEST_ONLY=(
  [hemit_vanilla_fm_joint_cfg]=test_hemit_vanilla_fm_joint_cfg_v2.sbatch
  [hemit_vanilla_fm_joint_cfg_v2]=test_hemit_vanilla_fm_joint_cfg_v2.sbatch
  [hemit_vanilla_fm_joint_film]=test_hemit_vanilla_fm_joint_film.sbatch
  [hemit_vanilla_fm_joint_film_v2]=test_hemit_vanilla_fm_joint_film_v2.sbatch
  [hemit_vanilla_fm_joint_seg]=test_hemit_vanilla_fm_joint_seg.sbatch
  [hemit_vanilla_fm_joint_init_cond_v2]=test_hemit_vanilla_fm_joint_init_cond.sbatch
  [hemit_vanilla_fm_joint_bridge_v2]=test_hemit_vanilla_fm_joint_bridge.sbatch
  [hemit_vanilla_fm_joint_gan]=test_hemit_vanilla_fm_joint_gan.sbatch
  [hemit_fm_tri_head_scratch]=test_hemit_fm_tri_head_scratch.sbatch
  [hemit_cond_fm_advanced_scratch]=test_hemit_cond_fm_advanced_scratch.sbatch
  [hemit_cond_fm_light_scratch]=test_hemit_cond_fm_light_scratch.sbatch
  [hemit_cond_fm_consistent_v2]=test_hemit_cond_fm_consistent_v2.sbatch
)

has_ckpt() {
  local n="$1"
  [[ -f "${ROOT}/checkpoints/${n}/${EPOCH}_net_G.pth" ]] && return 0
  [[ -f "${ROOT}/checkpoints/${n}_512/${EPOCH}_net_G.pth" ]] && return 0
  return 1
}

has_score() {
  local n="$1"
  [[ -f "${ROOT}/results/${n}/test_${EPOCH}/images/score.csv" ]] && return 0
  [[ -f "${ROOT}/results/${n}_512/test_${EPOCH}/images/score.csv" ]] && return 0
  return 1
}

submit() {
  local name="$1" sbatch_rel="$2"
  local sb="${ROOT}/bash_scripts/${sbatch_rel}"
  [[ -f "${sb}" ]] || { echo "SKIP ${name}: missing ${sbatch_rel}"; return; }
  if [[ "${DRY}" -eq 1 ]]; then
    echo "DRY  TEST_EPOCH=${EPOCH} sbatch bash_scripts/${sbatch_rel}  # ${name}"
  else
    echo "SUBMIT ${name} → ${sbatch_rel}"
    TEST_EPOCH="${EPOCH}" sbatch "${sb}"
  fi
}

echo "=== Batch eval @ epoch ${EPOCH} (dry_run=${DRY}) ==="

for name in "${!EVAL[@]}"; do
  has_ckpt "${name}" || continue
  has_score "${name}" && { echo "SKIP ${name}: score.csv already exists"; continue; }
  submit "${name}" "${EVAL[$name]}"
done

for name in "${!TEST_ONLY[@]}"; do
  has_ckpt "${name}" || continue
  has_score "${name}" && { echo "SKIP ${name}: score.csv already exists"; continue; }
  submit "${name}" "${TEST_ONLY[$name]}"
done

echo "Done. After jobs finish, copy score.csv files off cluster:"
echo "  scp 'cluster:~/Pix2pix_DualBranch/results/*/test_${EPOCH}/images/score.csv' ~/Downloads/"
