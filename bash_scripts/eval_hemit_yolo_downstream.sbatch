#!/bin/bash
# YOLO CD3 downstream: frozen detector on real vs generated test TIFFs.
#
#   export YOLO_WEIGHTS=weights/yolo_cd3_hemit.pt
#   export HEMIT_MANIFEST=eval/hemit/manifest.csv
#   sbatch bash_scripts/eval_hemit_yolo_downstream.sbatch
#
#SBATCH --job-name=yolo_cd3_eval
#SBATCH --output=logs/yolo_cd3_eval_%j.out
#SBATCH --error=logs/yolo_cd3_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs eval/hemit/yolo_downstream

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

pip install -q "numpy==1.22.4"
pip install -q "scikit-image>=0.19,<0.25" "tifffile>=2023.1"
pip install -q "ultralytics>=8.0" --no-deps
pip install -q opencv-python-headless pyyaml tqdm requests psutil py-cpuinfo pandas matplotlib seaborn pillow scipy

YOLO_WEIGHTS="${YOLO_WEIGHTS:-weights/yolo_cd3_hemit.pt}"
HEMIT_MANIFEST="${HEMIT_MANIFEST:-eval/hemit/manifest.csv}"
YOLO_OUTDIR="${YOLO_OUTDIR:-eval/hemit/yolo_downstream}"

[[ -f "${YOLO_WEIGHTS}" ]] || { echo "ERROR: train first — missing ${YOLO_WEIGHTS}" >&2; exit 1; }
[[ -f "${HEMIT_MANIFEST}" ]] || { echo "ERROR: missing ${HEMIT_MANIFEST}" >&2; exit 1; }

python scripts/run_hemit_yolo_downstream.py \
  --weights "${YOLO_WEIGHTS}" \
  --manifest "${HEMIT_MANIFEST}" \
  --outdir "${YOLO_OUTDIR}" \
  --imgsz "${YOLO_IMGSZ:-512}"

echo "Done. See ${YOLO_OUTDIR}/yolo_cd3_leaderboard.csv"
