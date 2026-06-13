#!/bin/bash
# Train YOLO CD3 detector on real HEMIT stains (train/val pseudo-labels).
#
#   export HEMIT_ROOT=/path/to/hemit   # official layout: train/label
#   # or: export DATAROOT=./datasets/hemit
#   sbatch bash_scripts/train_hemit_yolo_cd3.sbatch
#
#SBATCH --job-name=yolo_cd3_train
#SBATCH --output=logs/yolo_cd3_train_%j.out
#SBATCH --error=logs/yolo_cd3_train_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs datasets/hemit_cd3_yolo weights runs/hemit_yolo

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

# ultralytics may upgrade numpy→2.x; scikit-image<0.22 then breaks (np.bool8 removed)
pip install -q "scikit-image>=0.24" "ultralytics>=8.0"

YOLO_DATA="${YOLO_DATA:-datasets/hemit_cd3_yolo/hemit_cd3.yaml}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-weights/yolo_cd3_hemit.pt}"

if [[ ! -f "${YOLO_DATA}" ]]; then
  PREP_ARGS=(--outdir datasets/hemit_cd3_yolo)
  if [[ -n "${HEMIT_ROOT:-}" ]]; then
    PREP_ARGS+=(--hemit-root "${HEMIT_ROOT}")
  elif [[ -n "${DATAROOT:-}" ]]; then
    PREP_ARGS+=(--dataroot "${DATAROOT}")
  else
    echo "ERROR: set HEMIT_ROOT or DATAROOT, or pre-build ${YOLO_DATA}" >&2
    exit 1
  fi
  python scripts/hemit_yolo_prepare_cd3_dataset.py "${PREP_ARGS[@]}"
fi

python scripts/hemit_yolo_train_cd3.py \
  --data "${YOLO_DATA}" \
  --weights-out "${YOLO_WEIGHTS}" \
  --epochs "${YOLO_EPOCHS:-100}" \
  --imgsz "${YOLO_IMGSZ:-512}" \
  --batch "${YOLO_BATCH:-16}" \
  --device "${CUDA_VISIBLE_DEVICES:-0}"

echo "Done. Weights: ${YOLO_WEIGHTS}"
