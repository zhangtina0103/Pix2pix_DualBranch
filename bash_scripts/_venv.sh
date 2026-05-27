# Activate repo venv + CUDA libs. Source after bash_scripts/_common.sh.
VENV_DIR="${REPO_ROOT}/.venv-hemit"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "ERROR: ${VENV_DIR} missing. Run once:" >&2
  echo "  sbatch bash_scripts/setup_venv_engaging.sbatch" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
# shellcheck source=/dev/null
source "${REPO_ROOT}/bash_scripts/_cuda.sh"
