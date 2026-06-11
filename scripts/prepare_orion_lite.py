#!/usr/bin/env python3
"""
Build Orion-Lite pix2pix dataroot from MIPHEI-preprocessed ORION-CRC tiles.

Source layout (Zenodo ORIONCRC_dataset_tile_20x):
  train_dataframe.csv, val_dataframe.csv, test_dataframe.csv
  he/*.jpeg
  if/*.tiff   (multi-channel mIF, 16 markers)

Output (same as HEMIT pipeline):
  datasets/orion_lite/
    trainA/ trainB/  valA/ valB/  testA/ testB/
    meta.json
    train_manifest.csv   (subset rows used for train)

Usage:
  python scripts/prepare_orion_lite.py --src /path/to/ORIONCRC_dataset_tile_20x
  python scripts/prepare_orion_lite.py --src ... --dst ./datasets/orion_lite --n-train 1500
  python scripts/prepare_orion_lite.py --download --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

# MIPHEI / DiffVS ORION channel order (PD-1 omitted in training stacks).
ORION_CHANNEL_ORDER = [
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO",
    "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA",
]
# HEMIT-aligned 3-marker panel: DAPI/Hoechst, CD3/CD3e, panCK/Pan-CK
ORION_LITE_MARKERS = ["Hoechst", "CD3e", "Pan-CK"]
MARKER_TO_IDX = {name: ORION_CHANNEL_ORDER.index(name) for name in ORION_LITE_MARKERS}

ZENODO_ORION_URL = (
    "https://zenodo.org/records/15340874/files/ORIONCRC_dataset_tile_20x.zip?download=1"
)


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    # CSV paths may be he/foo.jpeg or just foo.jpeg
    if (root / p).exists():
        return root / p
    name = p.name
    for sub in ("he", "if", "nuclei"):
        cand = root / sub / name
        if cand.exists():
            return cand
    return root / p


def _slide_col(df: pd.DataFrame) -> str:
    for col in ("slide_name", "in_slide_name", "slide_id"):
        if col in df.columns:
            return col
    raise ValueError(f"No slide column in dataframe. Columns: {list(df.columns)}")


def _safe_stem(row: pd.Series, slide_col: str, idx: int) -> str:
    slide = str(row.get(slide_col, "slide"))
    slide = re.sub(r"[^\w.\-]+", "_", slide)
    he_name = Path(str(row["image_path"])).stem
    he_name = re.sub(r"[^\w.\-]+", "_", he_name)
    return f"{idx:06d}_{slide}_{he_name}"


def _resize_rgb(img: Image.Image, size: int) -> Image.Image:
    if img.size == (size, size):
        return img
    return img.resize((size, size), Image.Resampling.BILINEAR)


def _resize_label(arr: np.ndarray, size: int) -> np.ndarray:
    """arr: H×W×3 uint8/float."""
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    out = np.zeros((size, size, arr.shape[2]), dtype=arr.dtype)
    for c in range(arr.shape[2]):
        ch = Image.fromarray(arr[..., c])
        ch = ch.resize((size, size), Image.Resampling.BILINEAR)
        out[..., c] = np.asarray(ch)
    return out


def _load_label_stack(if_path: Path) -> np.ndarray:
    arr = tifffile.imread(if_path)
    if arr.ndim == 2:
        raise ValueError(f"Expected multi-channel mIF at {if_path}, got 2D")
    # H×W×C (MIPHEI / DiffVS)
    if arr.shape[-1] < len(ORION_CHANNEL_ORDER):
        # C×H×W fallback
        if arr.ndim == 3 and arr.shape[0] >= len(ORION_CHANNEL_ORDER):
            channels = [arr[idx] for idx in MARKER_TO_IDX.values()]
            stack = np.stack(channels, axis=-1)
        else:
            raise ValueError(f"Unexpected mIF shape {arr.shape} at {if_path}")
    else:
        stack = np.stack([arr[..., MARKER_TO_IDX[m]] for m in ORION_LITE_MARKERS], axis=-1)
    if stack.dtype != np.uint8:
        stack = np.clip(stack, 0, 255).astype(np.uint8)
    return stack


def sample_train_df(df: pd.DataFrame, n_train: int, seed: int) -> pd.DataFrame:
    slide_col = _slide_col(df)
    n_slides = df[slide_col].nunique()
    per_slide = max(1, n_train // n_slides)
    parts = []
    for _, group in df.groupby(slide_col, sort=True):
        k = min(per_slide, len(group))
        parts.append(group.sample(n=k, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n_train:
        out = out.sample(n=n_train, random_state=seed).sort_index()
    out = out.reset_index(drop=True)
    return out


def write_split(
    df: pd.DataFrame,
    src_root: Path,
    dst_root: Path,
    split: str,
    tile_size: int,
    symlink_he: bool,
) -> int:
    phase = split if split != "val" else "val"
    dir_a = dst_root / f"{phase}A"
    dir_b = dst_root / f"{phase}B"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    slide_col = _slide_col(df)
    n = 0
    for idx, row in df.iterrows():
        he_path = _resolve_path(src_root, str(row["image_path"]))
        if_path = _resolve_path(src_root, str(row["target_path"]))
        if not he_path.exists():
            print(f"  [warn] missing H&E: {he_path}", file=sys.stderr)
            continue
        if not if_path.exists():
            print(f"  [warn] missing mIF: {if_path}", file=sys.stderr)
            continue

        stem = _safe_stem(row, slide_col, idx)
        out_he = dir_a / f"{stem}.tif"
        out_if = dir_b / f"{stem}.tif"

        if symlink_he and tile_size == 256:
            # Fast path: symlink H&E if native 256²; always materialize labels (3ch extract).
            if not out_he.exists():
                out_he.symlink_to(he_path.resolve())
        else:
            he = _resize_rgb(Image.open(he_path).convert("RGB"), tile_size)
            he.save(out_he)

        label = _load_label_stack(if_path)
        label = _resize_label(label, tile_size)
        tifffile.imwrite(out_if, label)
        n += 1
    print(f"  {split}: {n} pairs → {dir_a.name}/ , {dir_b.name}/")
    return n


def find_orion_root(data_dir: Path) -> Path:
    for name in (
        "ORIONCRC_dataset_tile_20x",
        "ORION_dataset_20x",
        "orioncrc_dataset_tile_20x",
    ):
        cand = data_dir / name
        if (cand / "train_dataframe.csv").exists():
            return cand
    if (data_dir / "train_dataframe.csv").exists():
        return data_dir
    raise FileNotFoundError(
        f"No ORION tile root under {data_dir}. Expected train_dataframe.csv "
        f"(download from https://doi.org/10.5281/zenodo.15340874)"
    )


def download_orion(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ORIONCRC_dataset_tile_20x.zip"
    if not zip_path.exists():
        print(f"Downloading {ZENODO_ORION_URL} → {zip_path}")
        try:
            subprocess.run(
                ["aria2c", "-x", "16", "-s", "16", "-k", "10M", "-o", str(zip_path), ZENODO_ORION_URL],
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            import urllib.request
            print("aria2c unavailable; using urllib (single connection)...")
            urllib.request.urlretrieve(ZENODO_ORION_URL, zip_path)
    root = find_orion_root(data_dir)
    if (root / "train_dataframe.csv").exists() and (root / "he").is_dir():
        print(f"ORION tiles already extracted at {root}")
        return root
    print(f"Extracting {zip_path} ...")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(data_dir)], check=True)
    return find_orion_root(data_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Orion-Lite pix2pix dataroot from MIPHEI ORION tiles.")
    p.add_argument("--src", type=str, default=None, help="Path to ORIONCRC_dataset_tile_20x (with CSVs + he/ + if/)")
    p.add_argument("--dst", type=str, default="./datasets/orion_lite", help="Output dataroot (trainA/B, ...)")
    p.add_argument("--n-train", type=int, default=1500, help="Train tiles (stratified by slide); val/test kept full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile-size", type=int, default=512, help="Resize tiles (MIPHEI native=256; HEMIT protocol=512)")
    p.add_argument("--symlink-he", action="store_true", help="Symlink H&E at 256² (labels still extracted)")
    p.add_argument("--download", action="store_true", help="Download Zenodo zip into --data-dir first")
    p.add_argument("--data-dir", type=str, default="./data/orion", help="Download/extract parent when --download")
    args = p.parse_args()

    if args.download:
        src_root = download_orion(Path(args.data_dir).expanduser().resolve())
    elif args.src:
        src_root = find_orion_root(Path(args.src).expanduser().resolve())
    else:
        p.error("Provide --src or --download")

    dst_root = Path(args.dst).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"ORION src: {src_root}")
    print(f"pix2pix dst: {dst_root}")
    print(f"Markers: {ORION_LITE_MARKERS} (indices {[MARKER_TO_IDX[m] for m in ORION_LITE_MARKERS]})")

    train_df = pd.read_csv(src_root / "train_dataframe.csv")
    val_df = pd.read_csv(src_root / "val_dataframe.csv")
    test_df = pd.read_csv(src_root / "test_dataframe.csv")

    train_sub = sample_train_df(train_df, args.n_train, args.seed)
    manifest_path = dst_root / "train_manifest.csv"
    train_sub.to_csv(manifest_path, index=False)
    print(f"Train subset: {len(train_sub)} / {len(train_df)} rows (manifest → {manifest_path})")

    counts = {}
    for split, df in ("train", train_sub), ("val", val_df), ("test", test_df):
        counts[split] = write_split(
            df, src_root, dst_root, split, args.tile_size, symlink_he=args.symlink_he
        )

    meta = {
        "source": str(src_root),
        "markers": ORION_LITE_MARKERS,
        "marker_indices": MARKER_TO_IDX,
        "n_train_requested": args.n_train,
        "tile_size": args.tile_size,
        "seed": args.seed,
        "splits": counts,
        "test_count": counts["test"],
        "slide_split": "MIPHEI official (37 train slides / 2 val / 2 test)",
    }
    meta_path = dst_root / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"meta.json → {meta_path}")
    print(f"Done. {sum(counts.values())} pairs total. Use: DATAROOT={dst_root}")


if __name__ == "__main__":
    main()
