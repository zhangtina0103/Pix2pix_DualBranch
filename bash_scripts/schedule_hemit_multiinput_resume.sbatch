#!/bin/bash
# Schedule delayed resume for HEMIT multi-input he + dapi (two GPU jobs).
#
#   bash bash_scripts/schedule_hemit_multiinput_resume.sbatch
#   SLURM_BEGIN=now+8hours bash bash_scripts/schedule_hemit_multiinput_resume.sbatch
#
# Override begin time: SLURM_BEGIN accepts Slurm formats, e.g. now+6hours, now+360

set -euo pipefail
cd "$(dirname "$0")/.."

BEGIN="${SLURM_BEGIN:-now+6hours}"
echo "Scheduling hemit multi-input resume: he + dapi, begin=${BEGIN}"

for variant in he dapi; do
  echo "  queue ${variant} → resume_hemit_multiinput.sbatch"
  sbatch --begin="${BEGIN}" \
    --job-name="hemit_${variant}_res" \
    --export=ALL,HEMIT_MULTI_VARIANT="${variant}" \
    bash_scripts/resume_hemit_multiinput.sbatch
done

echo "Queued 2 jobs. Check: squeue -u \$USER"
