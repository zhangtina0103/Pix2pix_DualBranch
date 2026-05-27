# Load cluster CUDA and expose libs to conda PyTorch (source after: conda activate pix2pix).

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
elif [[ -f /usr/share/Modules/init/bash ]]; then
  # shellcheck source=/dev/null
  source /usr/share/Modules/init/bash
fi

if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  module load cuda/12.4 2>/dev/null || \
  module load cuda/12.2 2>/dev/null || \
  module load cuda/12.1 2>/dev/null || \
  module load cuda/11.8 2>/dev/null || \
  module load cuda 2>/dev/null || true
  module list 2>&1 || true
fi

_extra=()
[[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]] && _extra+=("${CUDA_HOME}/lib64")
[[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]] && _extra+=("${CONDA_PREFIX}/lib")
[[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib" ]] && \
  _extra+=("${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib")
[[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/lib" ]] && \
  _extra+=("${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/lib")
[[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib" ]] && \
  _extra+=("${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib")

if [[ ${#_extra[@]} -gt 0 ]]; then
  export LD_LIBRARY_PATH="$(IFS=:; echo "${_extra[*]}"):${LD_LIBRARY_PATH:-}"
fi
