# shellcheck shell=bash
# Sync GPU_IDS with GPUs visible to this job (not total GPUs on the node).
# Fixes stale login-shell `export GPU_IDS=0` and SLURM_GPUS_ON_NODE=2 when only 1 GPU allocated.

hemit_sync_gpu_env() {
  if [[ "${HEMIT_NO_GPU_SYNC:-0}" == "1" ]]; then
    export GPU_IDS="${GPU_IDS:-0}"
    return
  fi
  if [[ -n "${GPU_IDS:-}" ]]; then
    return
  fi
  local n=0
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    n="$(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0)"
  fi
  if [[ "${n}" -gt 0 ]]; then
    export GPU_IDS
    GPU_IDS="$(seq -s, 0 $((n - 1)))"
    export GPU_IDS
    echo "hemit_sync_gpu_env: visible_gpus=${n} -> GPU_IDS=${GPU_IDS}"
  else
    export GPU_IDS=0
  fi
}
