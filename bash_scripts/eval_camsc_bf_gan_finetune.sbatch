#!/bin/bash
# Eval HEMIT-init GAN finetune on CaMSC (pix2pix_ft / cut_ft / asp_ft).
#
#   export CAMSC_KFOLD_ROOT=~/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug
#   export CAMSC_ENABLE_AUG=1
#   CAMSC_MODEL=pix2pix_ft CAMSC_EVAL_FOLDS=0 TEST_EPOCH=110 sbatch bash_scripts/eval_camsc_bf_gan_finetune.sbatch
#SBATCH --job-name=camsc_gft_ev
#SBATCH --output=logs/camsc_gan_ft_eval_%j.out
#SBATCH --error=logs/camsc_gan_ft_eval_%j.err
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
source "${REPO_ROOT}/scripts/camsc_gan_finetune.sh"

export CAMSC_MODEL="${CAMSC_MODEL:-pix2pix_ft}"
case "${CAMSC_MODEL}" in
  pix2pix_ft|cut_ft|asp_ft) ;;
  *) echo "ERROR: CAMSC_MODEL=${CAMSC_MODEL}" >&2; exit 1 ;;
esac

export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
camsc_apply_train_data_env
camsc_apply_model_env

export MODE=test
export GPU_IDS=0
export TEST_EPOCH="${TEST_EPOCH:-$(camsc_gan_finetune_default_test_epoch)}"
export NUM_TEST="${NUM_TEST:-9999}"

FOLD_LIST="${CAMSC_EVAL_FOLDS:-0}"
NAME_SUFFIX="$(camsc_eval_name_suffix 0)"
echo "${CAMSC_MODEL} dirs: camsc_bf_${CAMSC_MODEL}_fold{N}${NAME_SUFFIX}"
echo "Eval folds: ${FOLD_LIST}  TEST_EPOCH=${TEST_EPOCH}"
echo ""

camsc_resolve_test_epoch() {
  local fold="$1"
  local dir="${REPO_ROOT}/checkpoints/$(camsc_resolved_train_name "${fold}")"
  if [[ ! -d "${dir}" ]]; then
    echo ""
    return 1
  fi
  local req="${REQUESTED_EPOCH:-${TEST_EPOCH}}"
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
    echo "[skip] fold ${FOLD}: no checkpoint"
    continue
  fi
  export TEST_EPOCH="${RESOLVED}"
  echo "========== test ${CAMSC_MODEL} fold ${FOLD} epoch=${TEST_EPOCH} =========="
  camsc_run_fold_test "${FOLD}"
  RAN_ANY=1
done

[[ "${RAN_ANY}" == "1" ]] || { echo "ERROR: no checkpoints" >&2; exit 1; }

OUT="${CAMSC_KFOLD_ROOT}/eval/${CAMSC_MODEL}"
mkdir -p "${OUT}"
python scripts/eval_camsc_kfold.py \
  --kfold-root "${CAMSC_KFOLD_ROOT}" \
  --results-root "${REPO_ROOT}/results" \
  --name-prefix "$(camsc_results_name_prefix)" \
  --name-suffix "${NAME_SUFFIX}" \
  --epoch "${REQUESTED_EPOCH}" \
  --k-folds "${CAMSC_KFOLDS}" \
  --out-dir "${OUT}"

echo "${CAMSC_MODEL} metrics: ${OUT}/kfold_aggregate.json"
echo "Compare fm_cross_attn_ft: ${CAMSC_KFOLD_ROOT}/eval/fm_cross_attn_ft/kfold_aggregate.json"
echo "Compare pix2pix scratch:  ${CAMSC_KFOLD_ROOT}/eval/pix2pix/kfold_aggregate.json"
