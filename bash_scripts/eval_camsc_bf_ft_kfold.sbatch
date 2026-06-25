#!/bin/bash
# Test all folds + extended metrics for one CaMSC finetuned model.
#
# FM cross-attn:
#   sbatch bash_scripts/eval_camsc_bf_ft_kfold.sbatch
# GAN:
#   CAMSC_GAN_FT_MODEL=cut_ft sbatch bash_scripts/eval_camsc_bf_ft_kfold.sbatch
#SBATCH --job-name=camsc_ft_ev
#SBATCH --output=logs/camsc_ft_eval_%j.out
#SBATCH --error=logs/camsc_ft_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/camsc_scratch_env.sh"

export CAMSC_KFOLD_ROOT="${CAMSC_KFOLD_ROOT:-${HOME}/orcd/scratch/camsc/datasets/camsc_bf_kfold_aug}"
export CAMSC_ENABLE_AUG="${CAMSC_ENABLE_AUG:-1}"
camsc_apply_train_data_env

# FM default; set CAMSC_GAN_FT_MODEL for GAN baselines
if [[ -n "${CAMSC_GAN_FT_MODEL:-}" ]]; then
  export CAMSC_MODEL="${CAMSC_GAN_FT_MODEL}"
else
  export CAMSC_MODEL=fm_cross_attn_ft
fi

case "${CAMSC_MODEL}" in
  fm_cross_attn_ft)
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/scripts/camsc_fm_finetune.sh"
    DEFAULT_EPOCH="$(camsc_finetune_default_test_epoch)"
    camsc_apply_model_env
    ;;
  pix2pix_ft|cut_ft|asp_ft|cyclegan_ft)
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/scripts/camsc_gan_finetune.sh"
    DEFAULT_EPOCH="$(camsc_gan_finetune_default_test_epoch)"
    camsc_apply_model_env
    ;;
  *)
    echo "ERROR: unknown model ${CAMSC_MODEL}" >&2
    exit 1
    ;;
esac

export MODE=test
export GPU_IDS=0
export TEST_EPOCH="${TEST_EPOCH:-${DEFAULT_EPOCH}}"
export NUM_TEST="${NUM_TEST:-9999}"
export FM_CHANNEL_WEIGHTS="${FM_CHANNEL_WEIGHTS:-1,1,0}"
export FM_STEPS="${FM_STEPS:-25}"

NAME_SUFFIX="$(camsc_eval_name_suffix 0)"
echo "Model=${CAMSC_MODEL}  epoch=${TEST_EPOCH}  suffix=${NAME_SUFFIX}"

camsc_resolve_test_epoch() {
  local fold="$1"
  local dir="${REPO_ROOT}/checkpoints/$(camsc_resolved_train_name "${fold}")"
  [[ -d "${dir}" ]] || return 1
  local req="${TEST_EPOCH}"
  [[ -f "${dir}/${req}_net_G.pth" ]] && echo "${req}" && return 0
  [[ -f "${dir}/latest_net_G.pth" ]] && echo "latest" && return 0
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
  return 1
}

REQUESTED_EPOCH="${TEST_EPOCH}"
RAN_ANY=0
for FOLD in $(seq 0 $((CAMSC_KFOLDS - 1))); do
  RESOLVED="$(camsc_resolve_test_epoch "${FOLD}" || true)"
  if [[ -z "${RESOLVED}" ]]; then
    echo "[skip] fold ${FOLD}: no checkpoint $(camsc_resolved_train_name "${FOLD}")"
    continue
  fi
  export TEST_EPOCH="${RESOLVED}"
  echo "========== test ${CAMSC_MODEL} fold ${FOLD} epoch=${TEST_EPOCH} =========="
  camsc_run_fold_test "${FOLD}"
  RAN_ANY=1
done

[[ "${RAN_ANY}" == "1" ]] || { echo "ERROR: no checkpoints for ${CAMSC_MODEL}" >&2; exit 1; }

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

echo "Done: ${OUT}/kfold_aggregate.json"
echo "      ${OUT}/leaderboard_main_metrics.csv"
