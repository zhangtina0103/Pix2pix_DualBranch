#!/bin/bash
# Eval one HEMIT multi-input variant @ TEST_EPOCH (default 80); ckpts/results on scratch.
#
#   HEMIT_MULTI_VARIANT=he_dapi TEST_EPOCH=80 sbatch bash_scripts/eval_hemit_multiinput.sbatch
#SBATCH --job-name=hemit_multi_ev
#SBATCH --output=logs/hemit_multiinput_eval_%j.out
#SBATCH --error=logs/hemit_multiinput_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODEL=vanilla_fm
export HEMIT_MULTI_VARIANT="${HEMIT_MULTI_VARIANT:?Set HEMIT_MULTI_VARIANT=he|dapi|he_dapi}"
export TEST_EPOCH="${TEST_EPOCH:-80}"
export NUM_TEST=945
export GPU_IDS=0
export DISPLAY_ID=-1

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/hemit_multiinput_env.sh"
hemit_multiinput_apply_env
export VANILLA_FM_ENV_LOCKED=1
source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"
export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
vanilla_fm_verify_locked_env
vanilla_fm_print_train_env

CKPT="${CHECKPOINTS_DIR}/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
echo "TRAIN_NAME=${TRAIN_NAME} TEST_EPOCH=${TEST_EPOCH}"
echo "DATAROOT=${DATAROOT}"
echo "CHECKPOINTS_DIR=${CHECKPOINTS_DIR} RESULTS_DIR=${RESULTS_DIR}"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}" >&2; exit 1; }

export MODE=test
bash scripts/run_hemit_vanilla_fm.sh
export MODE=metrics
bash scripts/run_hemit_vanilla_fm.sh

echo "Metrics: ${RESULTS_DIR}/${TRAIN_NAME}/test_${TEST_EPOCH}/score_cd3_panck.csv"
