#!/bin/bash
# Test + k-fold metrics for pix2pix only.
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   TEST_EPOCH=120 sbatch bash_scripts/eval_camsc_bf_pix2pix.sbatch
#SBATCH --job-name=camsc_p2p_ev
#SBATCH --output=logs/camsc_pix2pix_eval_%j.out
#SBATCH --error=logs/camsc_pix2pix_eval_%j.err
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

export CAMSC_MODEL=pix2pix
camsc_apply_model_env
export MODE=test
export GPU_IDS=0
export TEST_EPOCH="${TEST_EPOCH:-120}"
export NUM_TEST="${NUM_TEST:-9999}"

NAME_SUFFIX="_${HEMIT_TRAIN_SIZE:-512}"
[[ "${CAMSC_ENABLE_AUG:-0}" == "1" ]] && NAME_SUFFIX="${NAME_SUFFIX}_aug"

for FOLD in $(seq 0 $((CAMSC_KFOLDS - 1))); do
  CKPT="${REPO_ROOT}/checkpoints/camsc_bf_pix2pix_fold${FOLD}${NAME_SUFFIX}/${TEST_EPOCH}_net_G.pth"
  if [[ ! -f "${CKPT}" ]]; then
    echo "[skip] fold ${FOLD}: missing ${CKPT}"
    continue
  fi
  echo "========== test pix2pix fold ${FOLD} epoch=${TEST_EPOCH} =========="
  camsc_run_fold_test "${FOLD}"
done

OUT="${CAMSC_KFOLD_ROOT}/eval/pix2pix"
mkdir -p "${OUT}"
python scripts/eval_camsc_kfold.py \
  --kfold-root "${CAMSC_KFOLD_ROOT}" \
  --results-root "${REPO_ROOT}/results" \
  --name-prefix "$(camsc_results_name_prefix)" \
  --name-suffix "${NAME_SUFFIX}" \
  --epoch "${TEST_EPOCH}" \
  --k-folds "${CAMSC_KFOLDS}" \
  --out-dir "${OUT}"

echo "pix2pix metrics @ epoch ${TEST_EPOCH}: ${OUT}/kfold_aggregate.json"
