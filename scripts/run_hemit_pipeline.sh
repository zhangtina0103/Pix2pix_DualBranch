#!/usr/bin/env bash
# Entry point — prefer run_hemit_all.sh (multi-model).
exec "$(dirname "$0")/run_hemit_all.sh" "$@"
