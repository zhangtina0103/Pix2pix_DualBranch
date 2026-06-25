#!/bin/bash
# Test + metrics for pix2pix and fm_cross_attn only (vanilla FM → eval_camsc_kfold.sbatch).
#
#   TEST_EPOCH=50 sbatch bash_scripts/eval_camsc_bf_compare.sbatch
#SBATCH --job-name=camsc_cmp_ev
#SBATCH --output=logs/camsc_compare_eval_%j.out
#SBATCH --error=logs/camsc_compare_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/camsc_scratch_env.sh"

export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
camsc_apply_train_data_env

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"

export GPU_IDS=0
export TEST_EPOCH="${TEST_EPOCH:-50}"
export NUM_TEST="${NUM_TEST:-9999}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"

for CAMSC_MODEL in pix2pix fm_cross_attn; do
  export CAMSC_MODEL
  camsc_apply_model_env
  export MODE=test

  NAME_SUFFIX="$(camsc_eval_name_suffix 0)"
  [[ "${CAMSC_ENABLE_AUG:-0}" == "1" ]] || NAME_SUFFIX="_${HEMIT_TRAIN_SIZE:-512}"

  for FOLD in $(seq 0 $((CAMSC_KFOLDS - 1))); do
    echo "========== test ${CAMSC_MODEL} fold ${FOLD} epoch=${TEST_EPOCH} =========="
    camsc_run_fold_test "${FOLD}"
  done

  OUT="${CAMSC_KFOLD_ROOT}/eval/${CAMSC_MODEL}"
  mkdir -p "${OUT}"
  python scripts/eval_camsc_kfold.py \
    --kfold-root "${CAMSC_KFOLD_ROOT}" \
    --results-root "${REPO_ROOT}/results" \
    --name-prefix "$(camsc_results_name_prefix)" \
    --name-suffix "${NAME_SUFFIX}" \
    --epoch "${TEST_EPOCH}" \
    --k-folds "${CAMSC_KFOLDS}" \
    --out-dir "${OUT}"
done

echo "Compare summary:"
for m in pix2pix fm_cross_attn; do
  echo "  ${m}: ${CAMSC_KFOLD_ROOT}/eval/${m}/kfold_aggregate.json"
done
