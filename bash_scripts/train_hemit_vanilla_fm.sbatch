#!/bin/bash
# Vanilla FM (joint 3ch) — train only. After pull with PY_MODEL fix.
#
#   sbatch bash_scripts/train_hemit_vanilla_fm.sbatch
#   # tune size first (login node ok):
#   python scripts/count_fm_params.py --fm_channels 96,192,256 --fm_num_res_blocks 2
#
# Then test + metrics:
#   MODEL=vanilla_fm MODE=test sbatch bash_scripts/run_hemit_all.sbatch
#   MODEL=vanilla_fm MODE=metrics METRICS_COMPOSITE=1 sbatch bash_scripts/run_hemit_all.sbatch
#SBATCH --job-name=hemit_vfm
#SBATCH --output=logs/hemit_vanilla_fm_%j.out
#SBATCH --error=logs/hemit_vanilla_fm_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/virtual-staining/data/hemit}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
export FM_CHANNELS="${FM_CHANNELS:-104,208,288}"
export FM_RESBLOCKS="${FM_RESBLOCKS:-2}"
export FM_STEPS="${FM_STEPS:-50}"
export FM_LAMBDA_L1="${FM_LAMBDA_L1:-100}"
export DISPLAY_ID="${DISPLAY_ID:--1}"

bash scripts/run_hemit_vanilla_fm.sh
