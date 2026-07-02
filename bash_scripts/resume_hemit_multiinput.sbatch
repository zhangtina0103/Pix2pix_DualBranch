#!/bin/bash
# Resume HEMIT multi-input FM in 6h chunks until epoch 80.
#
#   HEMIT_MULTI_VARIANT=he sbatch bash_scripts/resume_hemit_multiinput.sbatch
#   HEMIT_MULTI_VARIANT=dapi RESUME_FROM_EPOCH=45 sbatch bash_scripts/resume_hemit_multiinput.sbatch
#   HEMIT_MULTI_VARIANT=he_dapi sbatch bash_scripts/resume_hemit_multiinput.sbatch
#
# Schedule he + dapi resume 6h from now:
#   bash bash_scripts/schedule_hemit_multiinput_resume.sbatch
#
# RESUME_FROM_EPOCH optional — auto-detects latest *_net_G.pth under scratch checkpoints.
#SBATCH --job-name=hemit_multi_res
#SBATCH --output=logs/hemit_multiinput_resume_%j.out
#SBATCH --error=logs/hemit_multiinput_resume_%j.err
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
export HEMIT_NO_GPU_SYNC=1
export GPU_IDS="${GPU_IDS:-0}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export HEMIT_MULTI_VARIANT="${HEMIT_MULTI_VARIANT:-he}"

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
bash "${REPO_ROOT}/scripts/resume_hemit_multiinput.sh"
