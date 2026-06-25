#!/bin/bash
# Test vanilla FM k-folds + aggregate metrics (separate from compare eval).
#
#   TEST_EPOCH=50 sbatch bash_scripts/eval_camsc_kfold.sbatch
#SBATCH --job-name=camsc_vfm_ev
#SBATCH --output=logs/camsc_kfold_eval_%j.out
#SBATCH --error=logs/camsc_kfold_eval_%j.err
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

export CAMSC_MODEL=vanilla_fm
export GPU_IDS=0
export TEST_EPOCH="${TEST_EPOCH:-50}"
export NUM_TEST="${NUM_TEST:-9999}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"

camsc_apply_model_env

for FOLD in $(seq 0 $((CAMSC_KFOLDS - 1))); do
  echo "========== test vanilla_fm fold ${FOLD} epoch=${TEST_EPOCH} =========="
  camsc_run_fold_test "${FOLD}"
done

OUT="${CAMSC_KFOLD_ROOT}/eval/vanilla_fm"
mkdir -p "${OUT}"
python scripts/eval_camsc_kfold.py \
  --kfold-root "${CAMSC_KFOLD_ROOT}" \
  --results-root "${REPO_ROOT}/results" \
  --name-prefix "$(camsc_results_name_prefix)" \
  --name-suffix "_${HEMIT_TRAIN_SIZE:-512}" \
  --epoch "${TEST_EPOCH}" \
  --k-folds "${CAMSC_KFOLDS}" \
  --out-dir "${OUT}"

echo "vanilla_fm metrics: ${OUT}/kfold_aggregate.json"
