#!/bin/bash
# Delayed CycleGAN-ft eval (single fold). Default: fold 0, start in 16 minutes.
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   bash bash_scripts/schedule_camsc_cyclegan_ft_eval_fold0.sbatch
#
# Override: CAMSC_EVAL_FOLDS=0 TEST_EPOCH=110 SLURM_BEGIN=now+30 bash ...
#SBATCH --job-name=ev_cg_ft0
#SBATCH --output=logs/camsc_cyclegan_ft_eval_%j.out
#SBATCH --error=logs/camsc_cyclegan_ft_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

set -euo pipefail
cd "$(dirname "$0")/.."

export CAMSC_KFOLD_ROOT="${CAMSC_KFOLD_ROOT:-${HOME}/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug}"
export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
export CAMSC_GAN_FT_MODEL=cyclegan_ft
export CAMSC_EVAL_FOLDS="${CAMSC_EVAL_FOLDS:-0}"
export TEST_EPOCH="${TEST_EPOCH:-110}"

BEGIN="${SLURM_BEGIN:-now+16}"
echo "Scheduling cyclegan_ft eval fold(s)=${CAMSC_EVAL_FOLDS} epoch=${TEST_EPOCH} begin=${BEGIN}"

sbatch --begin="${BEGIN}" -J ev_cg_ft0 \
  --export=ALL,CAMSC_KFOLD_ROOT,CAMSC_ENABLE_AUG,CAMSC_GAN_FT_MODEL,CAMSC_EVAL_FOLDS,TEST_EPOCH \
  bash_scripts/eval_camsc_bf_ft_kfold.sbatch

echo "Queued. Check: squeue -u \$USER"
