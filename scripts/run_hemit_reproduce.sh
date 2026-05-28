#!/usr/bin/env bash
# Shortcut: paper dual-branch (same as MODEL=dualbranch).
export MODEL="${MODEL:-dualbranch}"
exec "$(dirname "$0")/run_hemit_all.sh"
