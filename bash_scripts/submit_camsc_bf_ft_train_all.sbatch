#!/bin/bash
# Submit all CaMSC finetune k-fold training (folds 1-4 by default; fold 0 done).
# For folds 1-4 only: bash bash_scripts/submit_camsc_bf_ft_train_folds1to4.sbatch

set -euo pipefail
cd "$(dirname "$0")/.."
export CAMSC_KFOLD_ROOT="${CAMSC_KFOLD_ROOT:-${HOME}/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug}"
export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"

ARRAY="${SLURM_FOLD_ARRAY:-1-4}"

submit_gan() {
  local model="$1" jname="$2"
  if [[ "${FORCE:-0}" != "1" ]] && ls checkpoints/camsc_bf_${model}_fold4_512_aug_512/110_net_G.pth 2>/dev/null; then
    echo "[skip] ${model} (fold4 @110 exists; FORCE=1 to resubmit)"
    return 0
  fi
  echo "[submit] FINETUNE ${model} folds ${ARRAY}"
  CAMSC_GAN_FT_MODEL="${model}" sbatch --array="${ARRAY}" -J "${jname}" \
    bash_scripts/train_camsc_bf_gan_finetune_kfold_array.sbatch
}

if [[ "${FORCE:-0}" != "1" ]] && ls checkpoints/camsc_bf_fm_cross_attn_ft_fold4_512_aug_512/110_net_G.pth 2>/dev/null; then
  echo "[skip] fm_cross_attn_ft"
else
  echo "[submit] FINETUNE fm_cross_attn_ft folds ${ARRAY}"
  sbatch --array="${ARRAY}" -J xa_ft_kf \
    bash_scripts/train_camsc_bf_fm_cross_attn_finetune_kfold_array.sbatch
fi

submit_gan pix2pix_ft p2p_ft_kf
submit_gan cut_ft      cut_ft_kf
submit_gan asp_ft      asp_ft_kf
submit_gan cyclegan_ft cg_ft_kf

echo "Done. Monitor: squeue -u \$USER"
