#!/bin/bash
# Schedule delayed resume for HEMIT multi-input he + dapi (two rounds, 6h apart).
#
# Default: resume at now+6h and now+12h (4 jobs total: he + dapi each round).
#
#   bash bash_scripts/schedule_hemit_multiinput_resume.sbatch
#
# Custom offset / rounds:
#   SLURM_RESUME_OFFSET_HOURS=6 SLURM_RESUME_ROUNDS=3 bash bash_scripts/schedule_hemit_multiinput_resume.sbatch

set -euo pipefail
cd "$(dirname "$0")/.."

OFFSET_HOURS="${SLURM_RESUME_OFFSET_HOURS:-6}"
ROUNDS="${SLURM_RESUME_ROUNDS:-2}"

echo "Scheduling hemit multi-input resume: he + dapi"
echo "  rounds=${ROUNDS}  offset=${OFFSET_HOURS}h  → now+${OFFSET_HOURS}h .. now+$((ROUNDS * OFFSET_HOURS))h"

for round in $(seq 1 "${ROUNDS}"); do
  hours=$(( round * OFFSET_HOURS ))
  BEGIN="now+${hours}hours"
  echo ""
  echo "==> round ${round}/${ROUNDS} begin=${BEGIN}"
  for variant in he dapi; do
    echo "  queue ${variant} (round ${round})"
    sbatch --begin="${BEGIN}" \
      --job-name="hemit_${variant}_r${round}" \
      --export=ALL,HEMIT_MULTI_VARIANT="${variant}" \
      bash_scripts/resume_hemit_multiinput.sbatch
  done
done

echo ""
echo "Queued $((ROUNDS * 2)) jobs. Check: squeue -u \$USER --start"
