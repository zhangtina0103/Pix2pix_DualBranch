#!/usr/bin/env bash
# List every 0-byte file under HEMIT (train/val/test input+label).
#
#   bash scripts/audit_hemit_zero_bytes.sh
#   HEMIT_ROOT=/home/zhangtin/HEMIT bash scripts/audit_hemit_zero_bytes.sh
#   HEMIT_ROOT=~/HEMIT OUT=~/hemit_zero_bytes.txt bash scripts/audit_hemit_zero_bytes.sh
set -euo pipefail

HEMIT_ROOT="${HEMIT_ROOT:-${HOME}/HEMIT}"
OUT="${OUT:-}"

if [[ ! -d "${HEMIT_ROOT}" ]]; then
  echo "ERROR: HEMIT_ROOT not found: ${HEMIT_ROOT}" >&2
  exit 1
fi

echo "HEMIT_ROOT=${HEMIT_ROOT}"
echo ""

total=0
for split in train val test; do
  for sub in input label; do
    dir="${HEMIT_ROOT}/${split}/${sub}"
    if [[ ! -d "${dir}" ]]; then
      echo "${split}/${sub}: (missing)"
      continue
    fi
    n=$(find "${dir}" -type f -size 0 2>/dev/null | wc -l | tr -d ' ')
    all=$(find "${dir}" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "${split}/${sub}: ${n} zero-byte / ${all} files"
    total=$((total + n))
  done
done

echo ""
echo "TOTAL zero-byte: ${total}"

if (( total == 0 )); then
  echo "OK — no 0-byte files under ${HEMIT_ROOT}"
  exit 0
fi

echo ""
echo "Listing paths:"
_list() {
  find "${HEMIT_ROOT}" -type f -size 0 2>/dev/null | sort
}
if [[ -n "${OUT}" ]]; then
  _list | tee "${OUT}"
  echo ""
  echo "Wrote ${OUT}"
else
  _list
fi

exit 1
