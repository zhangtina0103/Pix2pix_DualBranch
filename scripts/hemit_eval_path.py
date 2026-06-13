"""Import hemit_eval from scripts/ (works without hemit.eval package on cluster)."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_hemit_eval_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return scripts
