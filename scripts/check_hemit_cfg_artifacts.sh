#!/usr/bin/env bash
# Audit CFG / FiLM artifacts on the cluster (run from repo root on GPU node).
#   bash scripts/check_hemit_cfg_artifacts.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== HEMIT CFG / FiLM artifact audit ==="
echo "REPO_ROOT=${ROOT}"
echo

names=(
  hemit_vanilla_fm_joint_perc
  hemit_vanilla_fm_joint_perc_512
  hemit_vanilla_fm_joint_cfg
  hemit_vanilla_fm_joint_cfg_v2
  hemit_vanilla_fm_joint_cfg_v2_512
  hemit_vanilla_fm_joint_film
  hemit_vanilla_fm_joint_film_v2
  hemit_vanilla_fm_joint_film_v2_512
  hemit_fm_cross_attn_scratch
  hemit_fm_cross_attn_scratch_512
)

for n in "${names[@]}"; do
  for d in "checkpoints/${n}" "checkpoints/${n}_512"; do
    [[ -d "${ROOT}/${d}" ]] || continue
    echo "--- ${d} ---"
    ls -1 "${ROOT}/${d}"/*_net_G.pth 2>/dev/null | tail -5 || echo "  (no G checkpoints)"
    latest=$(ls -1 "${ROOT}/${d}"/*_net_G.pth 2>/dev/null | sed 's/.*\///;s/_net_G.pth//' | sort -n | tail -1)
    [[ -n "${latest}" ]] && echo "  latest epoch: ${latest}"
  done
done

echo
echo "=== CFG score CSVs ==="
find "${ROOT}/results" -maxdepth 4 -name 'score_cfg_w*.csv' 2>/dev/null | sort || true
find "${ROOT}/results" -path '*joint_cfg*' -name 'score.csv' 2>/dev/null | sort || true

echo
echo "=== Recent CFG/FiLM SLURM logs ==="
ls -lt "${ROOT}/logs"/hemit_vfm_joint_cfg* "${ROOT}/logs"/hemit_vfm_joint_film* 2>/dev/null | head -15 || true

echo
echo "=== Seed checkpoint for re-finetune ==="
for seed in \
  checkpoints/hemit_vanilla_fm_joint_perc/80_net_G.pth \
  checkpoints/hemit_vanilla_fm_joint_perc_512/80_net_G.pth \
  checkpoints/hemit_fm_cross_attn_scratch/80_net_G.pth \
  checkpoints/hemit_fm_cross_attn_scratch_512/80_net_G.pth; do
  if [[ -f "${ROOT}/${seed}" ]]; then
    echo "  OK ${seed}"
  else
    echo "  -- missing ${seed}"
  fi
done
