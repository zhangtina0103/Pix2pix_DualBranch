"""Python path for HEMIT comparison models (in-repo, no vs_v2 dependency)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT: Path | None = None


def repo_root() -> Path:
    global _REPO_ROOT
    if _REPO_ROOT is None:
        _REPO_ROOT = Path(__file__).resolve().parent.parent
    return _REPO_ROOT


def ensure_repo_paths() -> Path:
    root = repo_root()
    hemit = root / "hemit"
    for p in (root, hemit, hemit / "training", hemit / "eval"):
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)
    return root
