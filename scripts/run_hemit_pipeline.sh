#!/usr/bin/env bash
# Wrapper — use run_hemit_reproduce.sh for official README reproduction.
exec "$(dirname "$0")/run_hemit_reproduce.sh" "$@"
