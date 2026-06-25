#!/bin/bash
# Submit eval (all 5 folds + extended metrics) for each finetuned model.
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   TEST_EPOCH=110 bash bash_scripts/submit_camsc_bf_ft_eval_all.sbatch

set -euo pipefail
cd "$(dirname "$0")/.."
export CAMSC_KFOLD_ROOT="${CAMSC_KFOLD_ROOT:-${HOME}/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug}"
export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
export TEST_EPOCH="${TEST_EPOCH:-110}"

echo "Eval epoch=${TEST_EPOCH}"

sbatch -J ev_xa_ft   bash_scripts/eval_camsc_bf_ft_kfold.sbatch

for m in pix2pix_ft cut_ft asp_ft cyclegan_ft; do
  CAMSC_GAN_FT_MODEL="${m}" TEST_EPOCH="${TEST_EPOCH}" sbatch -J "ev_${m}" bash_scripts/eval_camsc_bf_ft_kfold.sbatch
done

echo "Metrics under: ${CAMSC_KFOLD_ROOT}/eval/<model>/"
echo "  kfold_aggregate.json"
echo "  leaderboard_main_metrics.csv  (SSIM, Pearson, Spearman, PSNR, MAE, RMSE, LPIPS)"
