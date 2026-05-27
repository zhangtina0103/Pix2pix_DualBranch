# Shared setup for SLURM/bash entrypoints.
# Submit .sbatch from Pix2pix_DualBranch repo root:
#   sbatch bash_scripts/run_hemit_reproduce.sbatch

_submit="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${_submit}/bash_scripts/_common.sh" ]]; then
  REPO_ROOT="${_submit}"
elif [[ -f "${_submit}/_common.sh" ]]; then
  REPO_ROOT="$(cd "${_submit}/.." && pwd)"
else
  echo "ERROR: cannot find bash_scripts/_common.sh (submit from repo root?)" >&2
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
