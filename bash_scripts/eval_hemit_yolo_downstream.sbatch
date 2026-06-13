#!/bin/bash
# YOLO CD3 downstream: frozen detector on real vs generated test TIFFs.
#
# Uses separate env hemit_yolo — does NOT touch pix2pix_cuda.
#
#   export YOLO_WEIGHTS=weights/yolo_cd3_hemit.pt
#   export HEMIT_MANIFEST=eval/hemit/manifest.csv
#   export YOLO_REF_MODE=pseudo          # pseudo | yolo
#   export YOLO_CD3_POSITIVE_ONLY=1      # eval CD3+ tiles only (~26)
#   export YOLO_CONF=0.1
#   export YOLO_OUTDIR=eval/hemit/yolo_downstream_v2
#   sbatch bash_scripts/eval_hemit_yolo_downstream.sbatch
#
#SBATCH --job-name=yolo_cd3_eval
#SBATCH --output=logs/yolo_cd3_eval_%j.out
#SBATCH --error=logs/yolo_cd3_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs eval/hemit/yolo_downstream

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-hemit_yolo}" 2>/dev/null || {
  echo "ERROR: create hemit_yolo env first (see train_hemit_yolo_cd3.sbatch header)." >&2
  exit 1
}

YOLO_WEIGHTS="${YOLO_WEIGHTS:-weights/yolo_cd3_hemit.pt}"
HEMIT_MANIFEST="${HEMIT_MANIFEST:-eval/hemit/manifest.csv}"
YOLO_OUTDIR="${YOLO_OUTDIR:-eval/hemit/yolo_downstream}"
YOLO_REF_MODE="${YOLO_REF_MODE:-pseudo}"
YOLO_CONF="${YOLO_CONF:-0.1}"

EXTRA_ARGS=()
[[ "${YOLO_CD3_POSITIVE_ONLY:-1}" == "1" ]] && EXTRA_ARGS+=(--cd3-positive-only)

[[ -f "${YOLO_WEIGHTS}" ]] || { echo "ERROR: train first — missing ${YOLO_WEIGHTS}" >&2; exit 1; }
[[ -f "${HEMIT_MANIFEST}" ]] || { echo "ERROR: missing ${HEMIT_MANIFEST}" >&2; exit 1; }

python scripts/run_hemit_yolo_downstream.py \
  --weights "${YOLO_WEIGHTS}" \
  --manifest "${HEMIT_MANIFEST}" \
  --outdir "${YOLO_OUTDIR}" \
  --imgsz "${YOLO_IMGSZ:-512}" \
  --conf "${YOLO_CONF}" \
  --ref-mode "${YOLO_REF_MODE}" \
  "${EXTRA_ARGS[@]}"

echo "Done. See ${YOLO_OUTDIR}/yolo_cd3_leaderboard.csv"
