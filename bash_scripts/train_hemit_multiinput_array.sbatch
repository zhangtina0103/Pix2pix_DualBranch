#!/bin/bash
# Train all three HEMIT multi-input variants (array 0=he, 1=dapi, 2=he_dapi).
#
#   sbatch bash_scripts/train_hemit_multiinput_array.sbatch
#SBATCH --job-name=hemit_multi_arr
#SBATCH --output=logs/hemit_multiinput_train_%A_%a.out
#SBATCH --error=logs/hemit_multiinput_train_%A_%a.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-2

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

case "${SLURM_ARRAY_TASK_ID}" in
  0) export HEMIT_MULTI_VARIANT=he ;;
  1) export HEMIT_MULTI_VARIANT=dapi ;;
  2) export HEMIT_MULTI_VARIANT=he_dapi ;;
  *) echo "ERROR: bad array id ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

exec bash "${REPO_ROOT}/bash_scripts/train_hemit_multiinput.sbatch"
