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
# Force HEMIT dataroot — inherited HNSCC DATAROOT (4ch) breaks Normalize (output_nc=3).
unset DATAROOT
export DATAROOT="${HEMIT_DATAROOT:-${REPO_ROOT}/datasets/hemit}"
export HEMIT_SRC="${HEMIT_SRC:-/home/zhangtin/HEMIT}"
export HEMIT_TRAIN_SIZE="${HEMIT_TRAIN_SIZE:-512}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DISPLAY_ID=-1

python3 - <<'PY'
from pathlib import Path
import sys
from PIL import Image

root = Path("${DATAROOT}") / "trainB"
imgs = sorted(root.glob("*.tif")) + sorted(root.glob("*.tiff")) + sorted(root.glob("*.png"))
if not imgs:
    sys.exit(f"ERROR: no trainB images under {root}")
nc = len(Image.open(imgs[0]).getbands())
if nc != 3:
    sys.exit(
        f"ERROR: HEMIT trainB has {nc} channels (expected 3). "
        f"DATAROOT={root.parent} looks like HNSCC — unset DATAROOT or set HEMIT_DATAROOT=./datasets/hemit"
    )
print(f"OK: DATAROOT={root.parent} trainB channels={nc}")
PY

source "${REPO_ROOT}/bash_scripts/_vanilla_fm_sbatch_preamble.sh"
source "${REPO_ROOT}/scripts/vanilla_fm_env.sh"
vanilla_fm_apply_fm_cross_attn_scratch_cd3_env
unset FM_CROSS_ATTN_DECODER
export FM_CROSS_ATTN_HEADS="${FM_CROSS_ATTN_HEADS:-4}"
export VANILLA_FM_ENV_LOCKED=1
vanilla_fm_print_train_env
bash "${REPO_ROOT}/scripts/run_hemit_vanilla_fm.sh"
