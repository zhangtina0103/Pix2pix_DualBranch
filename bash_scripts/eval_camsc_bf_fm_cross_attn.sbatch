#!/bin/bash
# Test + metrics for fm_cross_attn only (skips folds without checkpoints).
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   CAMSC_EVAL_FOLDS=0 TEST_EPOCH=80 sbatch bash_scripts/eval_camsc_bf_fm_cross_attn.sbatch
#SBATCH --job-name=camsc_xa_ev
#SBATCH --output=logs/camsc_fm_cross_attn_eval_%j.out
#SBATCH --error=logs/camsc_fm_cross_attn_eval_%j.err
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
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"

export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
camsc_apply_train_data_env

export CAMSC_MODEL=fm_cross_attn
camsc_apply_model_env
export MODE=test
export GPU_IDS=0
export TEST_EPOCH="${TEST_EPOCH:-80}"
export NUM_TEST="${NUM_TEST:-9999}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"
export FM_STEPS="${FM_STEPS:-25}"

# Comma or space list, e.g. "0" or "0,1,2"
FOLD_LIST="${CAMSC_EVAL_FOLDS:-}"
if [[ -z "${FOLD_LIST}" ]]; then
  FOLD_LIST="$(seq 0 $((CAMSC_KFOLDS - 1)))"
fi

NAME_SUFFIX="$(camsc_eval_name_suffix 0)"
echo "fm_cross_attn checkpoint dirs: camsc_bf_fm_cross_attn_fold{N}${NAME_SUFFIX}"
echo "Eval folds: ${FOLD_LIST}  TEST_EPOCH=${TEST_EPOCH}  FM_STEPS=${FM_STEPS}"
echo ""

camsc_resolve_test_epoch() {
  local fold="$1"
  local dir="${REPO_ROOT}/checkpoints/$(camsc_resolved_train_name "${fold}")"
  if [[ ! -d "${dir}" ]]; then
    echo ""
    return 1
  fi
  local req="${REQUESTED_EPOCH:-${TEST_EPOCH}}"
  if [[ "${req}" == "latest" ]] && [[ -f "${dir}/latest_net_G.pth" ]]; then
    echo "latest"
    return 0
  fi
  if [[ -f "${dir}/${req}_net_G.pth" ]]; then
    echo "${req}"
    return 0
  fi
  if [[ -f "${dir}/latest_net_G.pth" ]]; then
    echo "latest"
    return 0
  fi
  local best="" f epoch
  shopt -s nullglob
  for f in "${dir}"/*_net_G.pth; do
    epoch="$(basename "${f}" _net_G.pth)"
    [[ "${epoch}" =~ ^[0-9]+$ ]] || continue
    if [[ -z "${best}" || "${epoch}" -gt "${best}" ]] && [[ "${epoch}" -le "${req}" ]]; then
      best="${epoch}"
    fi
  done
  shopt -u nullglob
  [[ -n "${best}" ]] && echo "${best}" && return 0
  echo ""
  return 1
}

RAN_ANY=0
REQUESTED_EPOCH="${TEST_EPOCH}"
for FOLD in ${FOLD_LIST}; do
  RESOLVED="$(camsc_resolve_test_epoch "${FOLD}" || true)"
  if [[ -z "${RESOLVED}" ]]; then
    echo "[skip] fold ${FOLD}: no checkpoint in $(camsc_resolved_train_name "${FOLD}")/"
    continue
  fi
  if [[ "${RESOLVED}" != "${REQUESTED_EPOCH}" ]]; then
    echo "[note] fold ${FOLD}: using epoch ${RESOLVED} (requested ${REQUESTED_EPOCH})"
  fi
  export TEST_EPOCH="${RESOLVED}"
  echo "========== test fm_cross_attn fold ${FOLD} name=$(camsc_resolved_train_name "${FOLD}") epoch=${TEST_EPOCH} =========="
  camsc_run_fold_test "${FOLD}"
  RAN_ANY=1
done

if [[ "${RAN_ANY}" != "1" ]]; then
  echo "ERROR: no fm_cross_attn checkpoints found for folds: ${FOLD_LIST}" >&2
  exit 1
fi

OUT="${CAMSC_KFOLD_ROOT}/eval/fm_cross_attn"
mkdir -p "${OUT}"
python scripts/eval_camsc_kfold.py \
  --kfold-root "${CAMSC_KFOLD_ROOT}" \
  --results-root "${REPO_ROOT}/results" \
  --name-prefix "$(camsc_results_name_prefix)" \
  --name-suffix "${NAME_SUFFIX}" \
  --epoch "${REQUESTED_EPOCH}" \
  --k-folds "${CAMSC_KFOLDS}" \
  --out-dir "${OUT}"

echo "fm_cross_attn metrics: ${OUT}/kfold_aggregate.json"
