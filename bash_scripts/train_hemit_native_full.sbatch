#!/bin/bash
# Train one native HEMIT baseline on full data (1024², joint 3ch). Max walltime 6h — re-submit resume if needed.
#
#   export HEMIT_SRC=/home/zhangtin/HEMIT
#   MODEL=pix2pix sbatch bash_scripts/train_hemit_native_full.sbatch
#   GPU_IDS is set from SLURM_GPUS_ON_NODE (ignores stale export GPU_IDS=0 on login node).
#   mit_normal_gpu: typically max 2 GPUs/job. BATCH_SIZE=4 default; try 8 if memory allows.
#   Override GPU list: HEMIT_NO_GPU_SYNC=1 GPU_IDS=0 sbatch ...
#   MODEL=pix2pixhd  sbatch bash_scripts/train_hemit_native_full.sbatch
#   MODEL=cut        sbatch bash_scripts/train_hemit_native_full.sbatch
#   MODEL=asp        sbatch bash_scripts/train_hemit_native_full.sbatch
#   MODEL=cyclegan   sbatch bash_scripts/train_hemit_native_full.sbatch
# If 6h times out before epoch 80: MODEL=pix2pix sbatch bash_scripts/resume_hemit_native_full.sbatch
#SBATCH --job-name=hemit_tr
#SBATCH --output=logs/hemit_train_%x_%j.out
#SBATCH --error=logs/hemit_train_%x_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
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
  export BATCH_SIZE="${BATCH_SIZE:-4}"
fi

echo "MODEL=${MODEL} MODE=${MODE} HEMIT_SRC=${HEMIT_SRC} GPU_IDS=${GPU_IDS} BATCH_SIZE=${BATCH_SIZE}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-?}"
ls "${DATAROOT}/trainA" | wc -l
bash scripts/run_hemit_all.sh
