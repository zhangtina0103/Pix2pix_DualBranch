#!/usr/bin/env bash
# Orion-Lite vanilla FM: train → test → metrics.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MODEL=vanilla_fm
export REPO_ROOT="${ROOT}"

source "${ROOT}/scripts/vanilla_fm_env.sh"
if [[ "${VANILLA_FM_ENV_LOCKED:-0}" != "1" ]]; then
  vanilla_fm_apply_orion_joint_perc_scratch_env
fi
source "${ROOT}/scripts/orion_model_profiles.sh"
vanilla_fm_verify_locked_env
vanilla_fm_print_train_env

exec bash "${ROOT}/scripts/run_orion_all.sh"
