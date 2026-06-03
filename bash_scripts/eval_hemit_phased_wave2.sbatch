#!/bin/bash
# Wave 2 eval (2 jobs): CycleGAN, vanilla FM joint_perc
#SBATCH --job-name=hemit_w2_ev
#SBATCH --output=logs/hemit_w2_eval_%A_%a.out
#SBATCH --error=logs/hemit_w2_eval_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
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
export TEST_EPOCH="${TEST_EPOCH:-80}"
export NUM_TEST=945
export GPU_IDS=0
export DISPLAY_ID=-1

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    export MODEL=cyclegan
    source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"
    CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
    echo "Wave2 eval: MODEL=cyclegan TRAIN_NAME=${TRAIN_NAME}"
    [[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}"; exit 1; }
    export MODE=test
    bash scripts/run_hemit_all.sh
    export MODE=metrics
    bash scripts/run_hemit_all.sh
    ;;
  1)
    source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
    vanilla_fm_apply_joint_perc_env
    export VANILLA_FM_ENV_LOCKED=1
    vanilla_fm_print_train_env
    CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
    echo "Wave2 eval: vanilla_fm TRAIN_NAME=${TRAIN_NAME}"
    [[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}"; exit 1; }
    export MODE=test
    bash scripts/run_hemit_vanilla_fm.sh
    export MODE=metrics
    bash scripts/run_hemit_vanilla_fm.sh
    ;;
  *) echo "ERROR: bad array id"; exit 1 ;;
esac
