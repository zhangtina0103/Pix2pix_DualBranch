#!/bin/bash
# Test then metrics (945 tiles). Prefer eval_hemit_native_full.sbatch (explicit two steps).
#
#   sbatch --array=1-5 bash_scripts/run_hemit_native_test_metrics_array.sbatch
#SBATCH --job-name=hemit_ev
#SBATCH --output=logs/hemit_eval_%A_%a.out
#SBATCH --error=logs/hemit_eval_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

MODELS=(dualbranch pix2pix pix2pixhd cut asp cyclegan)
export MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export TEST_EPOCH="${TEST_EPOCH:-80}"
export NUM_TEST=945
export GPU_IDS=0

echo "Task ${SLURM_ARRAY_TASK_ID}: MODEL=${MODEL} TEST_EPOCH=${TEST_EPOCH}"

echo "===== test ====="
export MODE=test
bash scripts/run_hemit_all.sh

echo "===== metrics ====="
export MODE=metrics
bash scripts/run_hemit_all.sh
