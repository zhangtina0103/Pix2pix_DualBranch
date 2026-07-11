#!/bin/bash
# Schedule eval (all 5 finetuned models, all folds) after training finishes.
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold
#   export CAMSC_ENABLE_AUG=1
#   bash bash_scripts/schedule_camsc_bf_ft_eval_all.sbatch
#
# Override delay: SLURM_BEGIN=now+2:00 bash bash_scripts/schedule_camsc_bf_ft_eval_all.sbatch

set -euo pipefail
cd "$(dirname "$0")/.."

export CAMSC_KFOLD_ROOT="${CAMSC_KFOLD_ROOT:-${HOME}/orcd/scratch/camsc/datasets/camsc_bf_kfold}"
export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
export TEST_EPOCH="${TEST_EPOCH:-110}"
BEGIN="${SLURM_BEGIN:-now+3:00}"

echo "Scheduling CaMSC finetune eval @ epoch ${TEST_EPOCH}"
echo "  CAMSC_KFOLD_ROOT=${CAMSC_KFOLD_ROOT}"
echo "  begin=${BEGIN}"

sbatch --begin="${BEGIN}" -J ev_xa_ft \
  --export=ALL,CAMSC_KFOLD_ROOT,CAMSC_ENABLE_AUG,TEST_EPOCH \
  bash_scripts/eval_camsc_bf_ft_kfold.sbatch

for m in pix2pix_ft cut_ft asp_ft cyclegan_ft; do
  CAMSC_GAN_FT_MODEL="${m}" sbatch --begin="${BEGIN}" -J "ev_${m}" \
    --export=ALL,CAMSC_KFOLD_ROOT,CAMSC_ENABLE_AUG,CAMSC_GAN_FT_MODEL,TEST_EPOCH \
    bash_scripts/eval_camsc_bf_ft_kfold.sbatch
done

echo "Queued 5 eval jobs (begin ${BEGIN}). Check: squeue -u \$USER"
echo "Metrics → ${CAMSC_KFOLD_ROOT}/eval/<model>/kfold_aggregate.json"
