#!/bin/bash
# Train YOLO CD3 detector on real HEMIT stains (train/val pseudo-labels).
#
# Uses separate env hemit_yolo — does NOT touch pix2pix_cuda.
# One-time setup (run once on login node):
#   conda create -n hemit_yolo python=3.10 -y
#   conda activate hemit_yolo
#   pip install torch torchvision ultralytics scikit-image tifffile scipy pandas pillow
#
#   export HEMIT_ROOT=/home/zhangtin/HEMIT
#   export FORCE_REPREP=0        # after prep succeeded once
#   sbatch bash_scripts/train_hemit_yolo_cd3.sbatch
#
# Bad GPU (CUDA ECC error): resubmit, or export YOLO_DEVICE=cpu YOLO_BATCH=8
#
#SBATCH --job-name=yolo_cd3_train
#SBATCH --output=logs/yolo_cd3_train_%j.out
#SBATCH --error=logs/yolo_cd3_train_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs datasets/hemit_cd3_yolo weights runs/hemit_yolo

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
# Heartbeat for SLURM (some clusters buffer stdout until job ends)
echo "yolo_cd3_train job=${SLURM_JOB_ID:-local} host=$(hostname) start=$(date)" \
  > "logs/yolo_cd3_train_${SLURM_JOB_ID:-local}.heartbeat"
log "job ${SLURM_JOB_ID:-local} starting on $(hostname)"
log "REPO_ROOT=${REPO_ROOT}"
log "cwd=$(pwd)"

for _conda in \
  "/orcd/home/002/zhangtin/miniforge3/etc/profile.d/conda.sh" \
  "/home/zhangtin/miniforge3/etc/profile.d/conda.sh" \
  "${HOME}/miniforge3/etc/profile.d/conda.sh"; do
  if [[ -f "${_conda}" ]]; then
    # shellcheck disable=SC1090
    source "${_conda}"
    break
  fi
done

conda activate "${CONDA_ENV:-hemit_yolo}" 2>/dev/null || {
  log "ERROR: create hemit_yolo env first (see header comments)."
  exit 1
}
log "conda env: ${CONDA_DEFAULT_ENV:-unknown}"
python -c "import sys; print('python', sys.executable)"

YOLO_DATA="${YOLO_DATA:-datasets/hemit_cd3_yolo/hemit_cd3.yaml}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-weights/yolo_cd3_hemit.pt}"

if [[ "${FORCE_REPREP:-1}" == "1" ]] || [[ ! -f "${YOLO_DATA}" ]]; then
  PREP_ARGS=(--outdir datasets/hemit_cd3_yolo --min-boxes 1 --bbox-pad 8 --min-box-px 24)
  if [[ -n "${HEMIT_ROOT:-}" ]]; then
    PREP_ARGS+=(--hemit-root "${HEMIT_ROOT}")
  elif [[ -n "${DATAROOT:-}" ]]; then
    PREP_ARGS+=(--dataroot "${DATAROOT}")
  else
    log "ERROR: set HEMIT_ROOT or DATAROOT, or pre-build ${YOLO_DATA}"
    exit 1
  fi
  log "preparing YOLO dataset: ${PREP_ARGS[*]}"
  python -u scripts/hemit_yolo_prepare_cd3_dataset.py "${PREP_ARGS[@]}"
  log "dataset prep done"
fi

log "starting YOLO train → ${YOLO_WEIGHTS} device=${YOLO_DEVICE:-0}"
python -u scripts/hemit_yolo_train_cd3.py \
  --data "${YOLO_DATA}" \
  --weights-out "${YOLO_WEIGHTS}" \
  --epochs "${YOLO_EPOCHS:-100}" \
  --imgsz "${YOLO_IMGSZ:-512}" \
  --batch "${YOLO_BATCH:-16}" \
  --device "${YOLO_DEVICE:-0}"

echo "Done. Weights: ${YOLO_WEIGHTS}"
log "finished OK"
