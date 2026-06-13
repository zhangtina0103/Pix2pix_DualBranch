#!/bin/bash
# Eval pix2pix @80 on Hoechst+CD3e+CD4 panel.
#
#   export TEST_EPOCH=80
#   sbatch bash_scripts/eval_orion_cd3_cd4_pix2pix.sbatch
#SBATCH --job-name=orion_cd4_p2p_ev
#SBATCH --output=logs/orion_cd3_cd4_pix2pix_eval_%j.out
#SBATCH --error=logs/orion_cd3_cd4_pix2pix_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs results

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-pix2pix_cuda}"
source "${REPO_ROOT}/bash_scripts/_cuda.sh"

source "${REPO_ROOT}/bash_scripts/_orion_cd3_cd4_preamble.sh"
_orion_cd3_cd4_check_dataroot

export MODEL=pix2pix
export TEST_EPOCH="${TEST_EPOCH:-80}"
export GPU_IDS=0

source "${REPO_ROOT}/scripts/orion_model_profiles.sh"
CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
echo "MODEL=pix2pix TRAIN_NAME=${TRAIN_NAME} epoch=${TEST_EPOCH} NUM_TEST=${NUM_TEST}"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}" >&2; exit 1; }

export MODE=test
bash scripts/run_orion_native.sh
export MODE=metrics
bash scripts/run_orion_native.sh

SCORE="results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/score.csv"
echo "score.csv → ${SCORE}"

python3 <<PY
import csv, statistics
from pathlib import Path
score = Path("${SCORE}")
rows = list(csv.DictReader(score.open()))
for col in ("cd3_pearson", "panck_pearson", "average_pearson", "average_ssim"):
    vals = [float(r[col]) for r in rows]
    print(f"mean {col}: {statistics.mean(vals):.4f}  (n={len(vals)})")
PY
