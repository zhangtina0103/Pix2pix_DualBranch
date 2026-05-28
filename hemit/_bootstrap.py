"""Import once at top of hemit/training/*.py and hemit/eval/*.py."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_hemit = _root / "hemit"
for _p in (_root, _hemit, _hemit / "training", _hemit / "eval"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
