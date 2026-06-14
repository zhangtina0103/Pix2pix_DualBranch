#!/bin/bash
# Cross-attn scratch + CD3 losses (focal L1, fg weight, velocity consistency). 1→80 from scratch.
# NOT fair vs baselines (1,2,1 + CD3 fg) — label separately in tables. Fair row: train_hemit_fm_cross_attn_scratch @80.
#
#   sbatch bash_scripts/train_hemit_fm_cross_attn_scratch_cd3.sbatch
#   export TEST_EPOCH=80 && sbatch bash_scripts/eval_hemit_fm_cross_attn_scratch_cd3.sbatch
#SBATCH --job-name=hemit_xa_cd3
#SBATCH --output=logs/hemit_fm_cross_attn_scratch_cd3_%j.out
#SBATCH --error=logs/hemit_fm_cross_attn_scratch_cd3_%j.err
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

export MODEL=vanilla_fm MODE=train
# This job always trains on HEMIT (3ch). Override with HEMIT_DATAROOT if needed.
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export DATAROOT="${HEMIT_DATAROOT:-${REPO_ROOT}/datasets/hemit}"
export HEMIT_TRAIN_SIZE="${HEMIT_TRAIN_SIZE:-512}"
export BATCH_SIZE="${BATCH_SIZE:-2}"  # vel consist = extra UNet pass; use 2 @512²
export DISPLAY_ID=-1

if [[ ! -d "${DATAROOT}/trainB" ]] || [[ -z "$(ls -A "${DATAROOT}/trainB" 2>/dev/null)" ]]; then
  echo "==> ${DATAROOT}/trainB missing or empty; preparing from HEMIT_SRC=${HEMIT_SRC}"
  MODEL=dualbranch MODE=prepare bash "${REPO_ROOT}/scripts/run_hemit_all.sh"
fi

python3 -c '
import os, sys
from pathlib import Path
from PIL import Image
dataroot = os.environ.get("DATAROOT", "")
if not dataroot:
    sys.exit("ERROR: DATAROOT not set")
root = Path(dataroot) / "trainB"
imgs = sorted(root.glob("*.tif")) + sorted(root.glob("*.tiff")) + sorted(root.glob("*.png"))
if not imgs:
    sys.exit(f"ERROR: no trainB under {root} — run MODE=prepare with HEMIT_SRC=/home/zhangtin/HEMIT")
nc = len(Image.open(imgs[0]).getbands())
if nc != 3:
    sys.exit(f"ERROR: trainB has {nc} channels (expected 3). DATAROOT={dataroot} is not HEMIT")
print(f"OK: DATAROOT={dataroot} trainB n={len(imgs)} channels={nc}")
'

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_fm_cross_attn_scratch_cd3_env
unset FM_CROSS_ATTN_DECODER
export FM_CROSS_ATTN_HEADS="${FM_CROSS_ATTN_HEADS:-4}"
export VANILLA_FM_ENV_LOCKED=1
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
