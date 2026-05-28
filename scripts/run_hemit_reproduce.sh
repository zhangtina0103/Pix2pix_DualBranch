#!/usr/bin/env bash
# Official HEMIT Dual-Branch (MODEL=dualbranch). Multi-model: scripts/run_hemit_all.sh
export MODEL="${MODEL:-dualbranch}"
exec "$(dirname "$0")/run_hemit_all.sh"
