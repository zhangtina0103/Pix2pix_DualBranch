#!/bin/bash
# Eval pix2pix screen @50; print GATE on mean cd3_pearson (column ch1 = CD3e).
#
#   export SCREEN_EPOCH=50
#   sbatch bash_scripts/eval_orion_cd3_cd4_pix2pix_screen.sbatch
#SBATCH --job-name=orion_cd4_gate
#SBATCH --output=logs/orion_cd3_cd4_pix2pix_gate_%j.out
#SBATCH --error=logs/orion_cd3_cd4_pix2pix_gate_%j.err
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
export TEST_EPOCH="${SCREEN_EPOCH:-50}"
export GPU_IDS=0

source "${REPO_ROOT}/scripts/orion_model_profiles.sh"
CKPT="${REPO_ROOT}/checkpoints/${TRAIN_NAME}/${TEST_EPOCH}_net_G.pth"
echo "MODEL=pix2pix TRAIN_NAME=${TRAIN_NAME} epoch=${TEST_EPOCH}"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}" >&2; exit 1; }

export MODE=test
bash scripts/run_orion_native.sh
export MODE=metrics
bash scripts/run_orion_native.sh

SCORE="results/${TRAIN_NAME}/test_${TEST_EPOCH}/images/score.csv"
[[ -f "${SCORE}" ]] || { echo "ERROR: missing ${SCORE}" >&2; exit 1; }

python3 <<PY
import pandas as pd
from pathlib import Path

score = Path("${SCORE}")
df = pd.read_csv(score)
# post_process.py: ch0=dapi(Hoechst), ch1=cd3(CD3e), ch2=panck(third marker = CD4 here)
cd3 = float(df["cd3_pearson"].mean())
ch2 = float(df["panck_pearson"].mean())
avg = float(df["average_pearson"].mean())
print(f"score.csv: {score}")
print(f"mean cd3_pearson (CD3e):  {cd3:.4f}")
print(f"mean ch2_pearson (CD4):    {ch2:.4f}")
print(f"mean average_pearson:      {avg:.4f}")
print()
if cd3 < 0.05:
    print("GATE PASS: pix2pix CD3 ~0 → run full fair sweep:")
    print("  sbatch bash_scripts/screen_orion_cd3_cd4_full.sbatch")
else:
    print("GATE FAIL: pix2pix already learns CD3 on this panel.")
    print("  Do not run FM sweep; Orion CRC still GAN-favorable for CD3.")
    print("  Optional backup panel: Hoechst,CD3e,PD-1 (re-prepare with ORION_MARKERS override).")
PY
