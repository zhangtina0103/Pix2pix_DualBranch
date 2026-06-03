#!/bin/bash
# Train one native HEMIT baseline on full data (1024², joint 3ch). Max walltime 6h — re-submit resume if needed.
#
#   export HEMIT_SRC=/home/zhangtin/HEMIT
#   MODEL=pix2pix sbatch bash_scripts/train_hemit_native_full.sbatch
#   Default: 1 GPU (#SBATCH --gres=gpu:1). For 2 GPUs: sbatch --gres=gpu:2 ...
#   GPU_IDS syncs to SLURM_GPUS_ON_NODE unless HEMIT_NO_GPU_SYNC=1.
#   Default HEMIT_TRAIN_SIZE=512. Checkpoints: hemit_*_512
#   MODEL=pix2pixhd  sbatch bash_scripts/train_hemit_native_full.sbatch
#   MODEL=cut        sbatch ...   # CUT/ASP default BATCH_SIZE=1 (PatchNCE OOM at 1024²)
#   MODEL=asp        sbatch bash_scripts/train_hemit_native_full.sbatch
#   MODEL=cyclegan   sbatch bash_scripts/train_hemit_native_full.sbatch
# If 6h times out before epoch 80: MODEL=pix2pix sbatch bash_scripts/resume_hemit_native_full.sbatch
#SBATCH --job-name=hemit_tr
#SBATCH --output=logs/hemit_train_%x_%j.out
#SBATCH --error=logs/hemit_train_%x_%j.err
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

export MODEL="${MODEL:?Set MODEL=pix2pix|pix2pixhd|cut|asp|cyclegan before sbatch}"
export MODE=train
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${DATAROOT:-./datasets/hemit}"
source "${REPO_ROOT}/bash_scripts/_hemit_gpu.sh"
if [[ "${HEMIT_NO_GPU_SYNC:-0}" != "1" ]]; then
  hemit_sync_gpu_env
else
  export GPU_IDS="${GPU_IDS:-0}"
fi

# BATCH_SIZE/TRAIN_NAME come from hemit_model_profiles.sh (not set until profile is sourced).
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hemit_model_profiles.sh"

echo "MODEL=${MODEL} MODE=${MODE} TRAIN_NAME=${TRAIN_NAME} HEMIT_SRC=${HEMIT_SRC} GPU_IDS=${GPU_IDS} BATCH_SIZE=${BATCH_SIZE}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-?}"
_n_train="$(ls -1 "${DATAROOT}/trainA" 2>/dev/null | wc -l | tr -d ' ')"
echo "trainA images: ${_n_train} (steps/epoch ≈ ceil(n / batch_size))"
bash scripts/run_hemit_all.sh
