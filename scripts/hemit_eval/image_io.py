"""Load HEMIT test.py image pairs (real_B / fake_B TIFF stacks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.io import imread

HEMIT_CHANNELS = ("dapi", "cd3", "panck")
HEMIT_CHANNEL_LABELS = ("Hoechst", "CD3e", "Pan-CK")


def resolve_image_dir(srcdir: str | Path) -> Path:
    """Accept test output dir or .../test_XX/images/."""
    srcdir = Path(srcdir).expanduser().resolve()
    images = srcdir / "images"
    if images.is_dir() and any(p.name.endswith("_fake_B.tif") for p in images.iterdir()):
        return images
    if any(p.name.endswith("_fake_B.tif") for p in srcdir.iterdir()):
        return srcdir
    raise FileNotFoundError(f"No *_fake_B.tif under {srcdir} or {srcdir}/images")


def list_fake_files(image_dir: Path) -> list[Path]:
    files = sorted(image_dir.glob("*_fake_B.tif"))
    if not files:
        raise FileNotFoundError(f"No *_fake_B.tif in {image_dir}")
    return files


def load_pair(fake_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (real, fake) float32 arrays (H,W,3) in [0,255] and base name."""
    if not fake_path.name.endswith("_fake_B.tif"):
        raise ValueError(f"Expected *_fake_B.tif, got {fake_path.name}")
    base = fake_path.name[:-11]
    real_path = fake_path.parent / f"{base}_real_B.tif"
    if not real_path.exists():
        real_path = fake_path.parent / f"{fake_path.name[:-10]}_real_B.tif"
    if not real_path.exists():
        raise FileNotFoundError(f"Missing real pair for {fake_path.name}")

    real = _to_float255(imread(real_path))
    fake = _to_float255(imread(fake_path))
    if real.shape != fake.shape:
        raise ValueError(f"Shape mismatch {real_path.name}: {real.shape} vs {fake.shape}")
    if real.ndim != 3 or real.shape[-1] < 3:
        raise ValueError(f"Expected (H,W,3) stack at {real_path}, got {real.shape}")
    return real[..., :3], fake[..., :3], base


def _to_float255(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() <= 1.0:
        arr = ((arr + 1.0) / 2.0) * 255.0
    return np.clip(arr, 0.0, 255.0)
