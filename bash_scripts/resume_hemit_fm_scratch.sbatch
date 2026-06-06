#!/bin/bash
# Resume FM in 6h chunks until profile end epoch.
# joint_perc @ 80: RESUME_PROFILE=joint_perc sbatch bash_scripts/resume_hemit_fm_scratch.sbatch
# scratch profiles @ 130: RESUME_PROFILE=seg_only_scratch|monai512|joint_perc_scratch|advanced|... sbatch ...
# MONAI: export BATCH_SIZE=2 (default profile bs=4 may OOM with mid-attn)
#
#   export RESUME_PROFILE=advanced
#   export RESUME_FROM_EPOCH=25   # optional; auto = latest numbered ckpt
#   sbatch bash_scripts/resume_hemit_fm_scratch.sbatch
#SBATCH --job-name=hemit_fm_res
#SBATCH --output=logs/hemit_fm_resume_%j.out
#SBATCH --error=logs/hemit_fm_resume_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs
source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

export DISPLAY_ID=-1
# BATCH_SIZE from hemit_model_profiles (4 @ 512²). OOM: BATCH_SIZE=2 sbatch ...
export DATAROOT="${DATAROOT:-./datasets/hemit}"

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
bash "${REPO_ROOT}/scripts/resume_hemit_fm_scratch.sh"
