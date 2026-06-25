#!/bin/bash
# Zero-shot: HEMIT cross-attn @80 on CaMSC test (no CaMSC training).
# Establishes transfer floor before finetune.
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   CAMSC_EVAL_FOLDS=0 sbatch bash_scripts/eval_camsc_bf_fm_cross_attn_zeroshot.sbatch
#SBATCH --job-name=camsc_xa_zs
#SBATCH --output=logs/camsc_fm_cross_attn_zeroshot_%j.out
#SBATCH --error=logs/camsc_fm_cross_attn_zeroshot_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/camsc_scratch_env.sh"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/camsc_fm_finetune.sh"

export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
camsc_apply_train_data_env

export CAMSC_MODEL=fm_cross_attn_zeroshot
camsc_apply_model_env
export MODE=test
export GPU_IDS=0
export TEST_EPOCH="${TEST_EPOCH:-${HEMIT_CROSS_ATTN_SRC_EPOCH:-80}}"
export NUM_TEST="${NUM_TEST:-9999}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"
export FM_STEPS="${FM_STEPS:-25}"

FOLD_LIST="${CAMSC_EVAL_FOLDS:-0}"
NAME_SUFFIX="$(camsc_eval_name_suffix 0)"
echo "HEMIT zero-shot ckpt: $(camsc_hemit_cross_attn_ckpt_path)"
echo "CaMSC dirs: camsc_bf_fm_cross_attn_zeroshot_fold{N}${NAME_SUFFIX}"
echo "Eval folds: ${FOLD_LIST}  TEST_EPOCH=${TEST_EPOCH}"
echo ""

RAN_ANY=0
for FOLD in ${FOLD_LIST}; do
  camsc_seed_hemit_cross_attn_ckpt "${FOLD}"
  echo "========== zero-shot test fold ${FOLD} epoch=${TEST_EPOCH} =========="
  camsc_run_fold_test "${FOLD}"
  RAN_ANY=1
done

if [[ "${RAN_ANY}" != "1" ]]; then
  echo "ERROR: no folds evaluated" >&2
  exit 1
fi

OUT="${CAMSC_KFOLD_ROOT}/eval/fm_cross_attn_zeroshot"
mkdir -p "${OUT}"
python scripts/eval_camsc_kfold.py \
  --kfold-root "${CAMSC_KFOLD_ROOT}" \
  --results-root "${REPO_ROOT}/results" \
  --name-prefix "$(camsc_results_name_prefix)" \
  --name-suffix "${NAME_SUFFIX}" \
  --epoch "${TEST_EPOCH}" \
  --k-folds "${CAMSC_KFOLDS}" \
  --out-dir "${OUT}"

echo "zero-shot metrics: ${OUT}/kfold_aggregate.json"
