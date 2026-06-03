# shellcheck shell=bash
# Sync GPU_IDS / BATCH_SIZE with the GPUs SLURM actually allocated.
# Fixes stale login-shell `export GPU_IDS=0` overriding sbatch defaults.

hemit_sync_gpu_env() {
  if [[ "${HEMIT_NO_GPU_SYNC:-0}" == "1" ]]; then
    export GPU_IDS="${GPU_IDS:-0}"
    export BATCH_SIZE="${BATCH_SIZE:-4}"
    return
  fi
  local n="${SLURM_GPUS_ON_NODE:-0}"
  if [[ -n "${SLURM_JOB_ID:-}" && "${n}" -gt 0 ]]; then
    export GPU_IDS
    GPU_IDS="$(seq -s, 0 $((n - 1)))"
    export GPU_IDS
    echo "hemit_sync_gpu_env: SLURM_GPUS_ON_NODE=${n} -> GPU_IDS=${GPU_IDS}"
  elif [[ -z "${GPU_IDS:-}" ]]; then
    export GPU_IDS=0
  fi
  export BATCH_SIZE="${BATCH_SIZE:-4}"
}
