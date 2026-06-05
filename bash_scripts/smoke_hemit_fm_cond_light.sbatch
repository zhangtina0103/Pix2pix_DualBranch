#!/bin/bash
# Fast smoke: pipeline check only (NOT for metric comparison).
#   ~500 patches, bs=4, 3 epochs → ~30–60 min total.
# OOM → BATCH_SIZE=2. Full-data smoke: unset MAX_DATASET_SIZE, N_EPOCHS=10.
#
#   sbatch bash_scripts/smoke_hemit_fm_cond_light.sbatch
#SBATCH --job-name=hemit_cfm_smoke
#SBATCH --output=logs/hemit_cond_fm_light_smoke_%j.out
#SBATCH --error=logs/hemit_cond_fm_light_smoke_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
echo "===== SMOKE START $(date -Is) job=${SLURM_JOB_ID:-local} host=$(hostname) pwd=$(pwd) ====="
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results
source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export DATAROOT="${DATAROOT:-./datasets/hemit}"
export DISPLAY_ID=-1
export NUM_THREADS=8
# Fast smoke defaults (override on cmdline if needed)
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MAX_DATASET_SIZE="${MAX_DATASET_SIZE:-500}"
export N_EPOCHS="${N_EPOCHS:-3}"
export N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-0}"
export SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-1}"
export VAL_FREQ="${VAL_FREQ:-99}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_cond_light_scratch_env
export VANILLA_FM_ENV_LOCKED=1
hemit_require_seg_masks "${DATAROOT}"

echo "===== SMOKE TRAIN (${N_EPOCHS} ep, bs=${BATCH_SIZE}, max_data=${MAX_DATASET_SIZE}) TRAIN_NAME=${TRAIN_NAME} ====="
vanilla_fm_print_train_env
export MODEL=vanilla_fm MODE=train
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"

export TEST_EPOCH="${TEST_EPOCH:-${N_EPOCHS}}"
export NUM_TEST=10
CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
[[ -f "${CKPT}" ]] || { echo "ERROR: smoke train did not write ${CKPT}" >&2; exit 1; }

echo "===== SMOKE TEST (${NUM_TEST} tiles) ====="
export MODEL=vanilla_fm MODE="test|metrics"
bash "${REPO_ROOT}/scripts/run_hemit_all.sh"
echo "Smoke done. Check results under checkpoints/${TRAIN_NAME}/web/ and score csv."
