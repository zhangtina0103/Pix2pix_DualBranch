#!/usr/bin/env bash
# Audit which HEMIT ablation checkpoints exist at epoch 80 (cluster).
# Run on GPU node from repo root:
#   bash scripts/audit_hemit_ablation_epoch80.sh
#   bash scripts/audit_hemit_ablation_epoch80.sh | tee logs/ablation_epoch80_audit.txt
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EPOCH="${TEST_EPOCH:-80}"

# All ablation / variant TRAIN_NAMEs (512² unless _full / _512 suffix on disk).
NAMES=(
  hemit_vanilla_fm_joint_perc
  hemit_vanilla_fm_joint_perc_full
  hemit_fm_cross_attn_scratch
  hemit_cond_fm_seg_only_scratch
  hemit_cond_fm_seg_init_scratch
  hemit_fm_tri_head_scratch
  hemit_cond_fm_advanced_scratch
  hemit_cond_fm_light_scratch
  hemit_cond_fm_consistent_v2
  hemit_vanilla_fm_joint_cfg
  hemit_vanilla_fm_joint_cfg_v2
  hemit_vanilla_fm_joint_film
  hemit_vanilla_fm_joint_film_v2
  hemit_vanilla_fm_joint_seg
  hemit_vanilla_fm_joint_init_cond_v2
  hemit_vanilla_fm_joint_bridge_v2
  hemit_vanilla_fm_joint_gan
  hemit_vanilla_fm_decoder_only
  hemit_vanilla_fm_perc_strong_scratch
  hemit_vanilla_fm_joint_perc_res3_scratch
  hemit_fm_cross_attn_scratch_focal
  hemit_fm_cross_attn_scratch_focal_tuned
  hemit_fm_cross_attn_scratch_focal_g075
  hemit_fm_cross_attn_scratch_focal_tuned_vel
  hemit_fm_cross_attn_scratch_focal_cd3
  hemit_fm_cross_attn_scratch_cd3
  hemit_fm_cross_attn_scratch_fg
  hemit_fm_cross_attn_scratch_vel
  hemit_fm_cross_attn_patchnce_scratch
  hemit_fm_cross_attn_init_scratch
)

have_ckpt=0
missing_ckpt=0
have_score=0
missing_score=0

printf '%-45s %-8s %-8s %s\n' "TRAIN_NAME" "ckpt@${EPOCH}" "score@${EPOCH}" "notes"
printf '%-45s %-8s %-8s %s\n' "----------" "--------" "----------" "-----"

for n in "${NAMES[@]}"; do
  ckpt=""
  score=""
  note=""
  for d in "checkpoints/${n}" "checkpoints/${n}_512"; do
    c="${ROOT}/${d}/${EPOCH}_net_G.pth"
    [[ -f "${c}" ]] && ckpt="YES" && note="${d}"
  done
  for r in \
    "results/${n}/test_${EPOCH}/images/score.csv" \
    "results/${n}_512/test_${EPOCH}/images/score.csv"; do
    [[ -f "${ROOT}/${r}" ]] && score="YES"
  done
  [[ -z "${ckpt}" ]] && ckpt="NO" && ((missing_ckpt++)) || ((have_ckpt++))
  if [[ "${score}" == "YES" ]]; then
    ((have_score++))
  else
    score="NO"
    ((missing_score++))
  fi
  printf '%-45s %-8s %-8s %s\n' "${n}" "${ckpt}" "${score}" "${note}"
done

echo
echo "=== Summary @ epoch ${EPOCH} ==="
echo "  checkpoints present: ${have_ckpt}/${#NAMES[@]}"
echo "  score.csv on cluster: ${have_score}/${#NAMES[@]}"
echo
echo "=== Local Downloads (if you synced score.csv) ==="
DL="${HOME}/Downloads"
for f in score-segonly80.csv score-2.csv score-3.csv score-4.csv score-5.csv score-6.csv score-7.csv score-vf-130.csv; do
  if [[ -f "${DL}/${f}" ]]; then
    echo "  OK  ${DL}/${f}"
  else
    echo "  --  ${DL}/${f}"
  fi
done
if [[ -d "${DL}/extended/cross_attn" ]]; then
  echo "  OK  ${DL}/extended/ (cross_attn, vanilla_fm, baselines @80)"
fi
echo
echo "Re-eval one run (example):"
echo "  export TEST_EPOCH=${EPOCH} && sbatch bash_scripts/eval_hemit_fm_cross_attn_scratch.sbatch"
echo "Batch re-eval all runs that have ckpt but no score:"
echo "  bash scripts/batch_eval_hemit_ablation_epoch80.sh"
