# Load cluster CUDA and expose libs to PyTorch (source after venv or conda activate).

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then
  # shellcheck source=/dev/null
  source /usr/share/Modules/init/bash
fi

if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  module load cuda/13.1.0 2>/dev/null || \
  module load cuda/12.4 2>/dev/null || \
  module load cuda/12.2 2>/dev/null || \
  module load cuda/12.1 2>/dev/null || \
  module load cuda/11.8 2>/dev/null || \
  module load cuda 2>/dev/null || true
  module list 2>&1 || true
fi

_py_prefix="${VIRTUAL_ENV:-${CONDA_PREFIX:-}}"

_extra=()
[[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]] && _extra+=("${CUDA_HOME}/lib64")
[[ -n "${_py_prefix}" && -d "${_py_prefix}/lib" ]] && _extra+=("${_py_prefix}/lib")
[[ -n "${_py_prefix}" && -d "${_py_prefix}/lib/python3.10/site-packages/nvidia/cublas/lib" ]] && \
  _extra+=("${_py_prefix}/lib/python3.10/site-packages/nvidia/cublas/lib")
[[ -n "${_py_prefix}" && -d "${_py_prefix}/lib/python3.10/site-packages/nvidia/cudnn/lib" ]] && \
  _extra+=("${_py_prefix}/lib/python3.10/site-packages/nvidia/cudnn/lib")
[[ -n "${_py_prefix}" && -d "${_py_prefix}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib" ]] && \
  _extra+=("${_py_prefix}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib")

if [[ ${#_extra[@]} -gt 0 ]]; then
  export LD_LIBRARY_PATH="$(IFS=:; echo "${_extra[*]}"):${LD_LIBRARY_PATH:-}"
fi

# Fail fast on bad GPU (ECC errors) before loading VGG / UNet.
hemit_gpu_smoke_test() {
  python - <<'PY'
import socket, sys
import torch

host = socket.gethostname()
if not torch.cuda.is_available():
    print(f"ERROR: no CUDA on {host}", file=sys.stderr)
    sys.exit(1)
try:
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    x = torch.randn(4096, 4096, device="cuda")
    _ = x @ x
    torch.cuda.synchronize()
    print(f"GPU OK: {name} (device {dev}) on {host}")
except RuntimeError as e:
    print(f"ERROR: GPU hardware failure on {host}: {e}", file=sys.stderr)
    print(f"Resubmit excluding this node: sbatch --exclude={host} ...", file=sys.stderr)
    sys.exit(1)
PY
}
