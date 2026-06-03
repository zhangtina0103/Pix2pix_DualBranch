#!/bin/bash
# Vanilla FM joint_perc (x1 + perc 0.1) on FULL HEMIT — scratch 1→130.
#
#   export HEMIT_SRC=/home/zhangtin/HEMIT_full
#   sbatch bash_scripts/prepare_hemit_full.sbatch
#   sbatch bash_scripts/train_hemit_vanilla_fm_joint_perc_full.sbatch
#   MODEL=vanilla_fm MODE=test|metrics TEST_EPOCH=130 sbatch bash_scripts/run_hemit_all.sbatch
#SBATCH --job-name=hemit_vfm_jp
#SBATCH --output=logs/hemit_vfm_joint_perc_full_%j.out
#SBATCH --error=logs/hemit_vfm_joint_perc_full_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT_full}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export BATCH_SIZE=1
export DISPLAY_ID=-1

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
vanilla_fm_apply_joint_perc_fulldata_env
export VANILLA_FM_ENV_LOCKED=1
vanilla_fm_print_train_env
bash scripts/run_hemit_vanilla_fm.sh
