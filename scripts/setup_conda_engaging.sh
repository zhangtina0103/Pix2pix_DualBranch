#!/usr/bin/env bash
# Run on interactive GPU: srun --partition=mit_normal_gpu --gres=gpu:1 --pty bash
#   cd ~/Pix2pix_DualBranch && bash scripts/setup_conda_engaging.sh
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_pix2pix_env.sh
