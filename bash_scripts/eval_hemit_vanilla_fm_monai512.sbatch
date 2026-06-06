#!/bin/bash
# monai512: test @ epoch then metrics (945 tiles).
#
#   export TEST_EPOCH=130
#   sbatch bash_scripts/eval_hemit_vanilla_fm_monai512.sbatch
#SBATCH --job-name=hemit_monai512_ev
#SBATCH --output=logs/hemit_monai512_eval_%j.out
#SBATCH --error=logs/hemit_monai512_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export TEST_EPOCH="${TEST_EPOCH:-130}"
export NUM_TEST="${NUM_TEST:-945}"
export GPU_IDS=0
export DISPLAY_ID=-1

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_monai512_env
export VANILLA_FM_ENV_LOCKED=1
export MODEL=vanilla_fm
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"
export VANILLA_FM_EXPECTED_TRAIN_NAME="${TRAIN_NAME}"
vanilla_fm_verify_locked_env
vanilla_fm_print_train_env

CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
echo "TRAIN_NAME=${TRAIN_NAME} TEST_EPOCH=${TEST_EPOCH}"
[[ -f "${CKPT}" ]] || { echo "ERROR: train first: sbatch bash_scripts/train_hemit_vanilla_fm_monai512.sbatch"; exit 1; }

echo "===== test ====="
export MODE=test
bash scripts/run_hemit_vanilla_fm.sh

echo "===== metrics ====="
export MODE=metrics
bash scripts/run_hemit_vanilla_fm.sh

echo "Done. See results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/score.csv"
