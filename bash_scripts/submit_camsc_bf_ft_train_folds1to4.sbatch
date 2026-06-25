#!/bin/bash
# Submit finetune for folds 1-4 only (fold 0 already done for all models).
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   bash bash_scripts/submit_camsc_bf_ft_train_folds1to4.sbatch

set -euo pipefail
cd "$(dirname "$0")/.."
export CAMSC_KFOLD_ROOT="${CAMSC_KFOLD_ROOT:-${HOME}/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug}"
export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
ARRAY="${SLURM_FOLD_ARRAY:-1-4}"

submit_gan() {
  local model="$1" jname="$2"
  echo "[submit] FINETUNE ${model} folds ${ARRAY}"
  CAMSC_GAN_FT_MODEL="${model}" sbatch --array="${ARRAY}" -J "${jname}" \
    bash_scripts/train_camsc_bf_gan_finetune_kfold_array.sbatch
}

echo "[submit] FINETUNE fm_cross_attn_ft folds ${ARRAY}"
sbatch --array="${ARRAY}" -J xa_ft_kf \
  bash_scripts/train_camsc_bf_fm_cross_attn_finetune_kfold_array.sbatch

submit_gan pix2pix_ft p2p_ft_kf
submit_gan cut_ft      cut_ft_kf
submit_gan asp_ft      asp_ft_kf
submit_gan cyclegan_ft cg_ft_kf

echo "Done (finetune folds ${ARRAY}). Check dirs: camsc_bf_*_ft_fold{N}_512_aug_512 with 80_net_* seeded + epochs 85..110"
echo "Monitor: squeue -u \$USER"
