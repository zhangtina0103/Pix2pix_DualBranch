#!/bin/bash
# Build HNSCC pix2pix dataroot from extracted tiles or ~/Downloads/hnscc.zip
#
#   sbatch bash_scripts/prepare_hnscc.sbatch
#SBATCH --job-name=hnscc_prep
#SBATCH --output=logs/hnscc_prepare_%j.out
#SBATCH --error=logs/hnscc_prepare_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/bash_scripts/_common.sh"
mkdir -p logs

source /home/zhangtin/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "${CONDA_ENV:-pix2pix_cuda}" 2>/dev/null || true

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/hnscc_scratch_env.sh"

export MODE=prepare
bash scripts/run_hnscc_all.sh

python3 -c "import json; m=json.load(open('${DATAROOT}/meta.json')); print(m)"
echo "HNSCC ready: DATAROOT=${DATAROOT}"
echo "Next: sbatch bash_scripts/run_hnscc_full_pipeline.sbatch"
