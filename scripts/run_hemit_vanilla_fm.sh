#!/usr/bin/env bash
# Vanilla flow matching: train.py → test.py → post_process.py (joint 3ch).
#
# Train: mentor flow_matching.py (x1+perc) or flow_matching_v.py (velocity). MONAI UNet ~11M params.
# Fresh train after recipe/arch change:
#   rm -rf checkpoints/hemit_vanilla_fm_joint
#   sbatch bash_scripts/train_hemit_vanilla_fm.sbatch
#
# Slow experimental train (ODE in loop): FM_USE_ODE_TRAIN=1 sbatch ...
#
# Cluster: sbatch bash_scripts/train_hemit_vanilla_fm.sbatch
# Avoid `sbatch --export=ALL` unless you know your shell has no stale FM_* vars.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MODEL=vanilla_fm

# shellcheck source=/dev/null
source "${ROOT}/scripts/vanilla_fm_env.sh"
# Phase sbatch sets apply_*_env + VANILLA_FM_ENV_LOCKED=1 before calling this script.
if [[ "${VANILLA_FM_ENV_LOCKED:-0}" != "1" ]]; then
  vanilla_fm_apply_train_env
fi
# shellcheck source=/dev/null
source "${ROOT}/scripts/hemit_model_profiles.sh"
vanilla_fm_verify_locked_env

if [[ "${PY_MODEL}" != "vanilla_fm" ]]; then
  echo "ERROR: expected PY_MODEL=vanilla_fm, got PY_MODEL=${PY_MODEL}" >&2
  exit 1
fi
vanilla_fm_print_train_env
echo "  PY_MODEL=${PY_MODEL}  TRAIN_NAME=${TRAIN_NAME}"

exec bash "${ROOT}/scripts/run_hemit_all.sh"
