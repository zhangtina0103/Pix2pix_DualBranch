#!/usr/bin/env bash
# Deprecated: use pip venv instead (matches README).
#   bash scripts/setup_venv_engaging.sh
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/setup_venv_engaging.sh
