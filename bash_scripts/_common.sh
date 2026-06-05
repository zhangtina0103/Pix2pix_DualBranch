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

# Skip generate_hemit_seg_masks.py when train/val/test Seg dirs match A counts.
# FORCE_HEMIT_SEG=1 to always regenerate.
hemit_seg_masks_ready() {
  local dataroot="$1"
  local split n_a n_seg
  for split in train val test; do
    [[ -d "${dataroot}/${split}A" && -d "${dataroot}/${split}Seg" ]] || return 1
    n_a=$(find "${dataroot}/${split}A" -maxdepth 1 -type f \
      \( -iname '*.tif' -o -iname '*.tiff' -o -iname '*.png' \) 2>/dev/null | wc -l | tr -d ' ')
    n_seg=$(find "${dataroot}/${split}Seg" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
    [[ "${n_a}" -gt 0 && "${n_seg}" -ge "${n_a}" ]] || return 1
  done
  return 0
}

hemit_maybe_generate_seg_masks() {
  local dataroot="${1:-./datasets/hemit}"
  dataroot="${REPO_ROOT}/${dataroot#./}"
  if [[ "${FORCE_HEMIT_SEG:-0}" == "1" ]]; then
    echo "FORCE_HEMIT_SEG=1: regenerating seg masks"
    python "${REPO_ROOT}/scripts/generate_hemit_seg_masks.py" --dataroot "${dataroot}"
  elif hemit_seg_masks_ready "${dataroot}"; then
    echo "Seg masks ready under ${dataroot} — skipping generate_hemit_seg_masks.py (FORCE_HEMIT_SEG=1 to regen)"
  else
    python "${REPO_ROOT}/scripts/generate_hemit_seg_masks.py" --dataroot "${dataroot}"
  fi
}
