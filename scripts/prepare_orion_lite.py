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
  python scripts/prepare_orion_lite.py --src ... --dst ./datasets/orion_lite --n-train 1500 --n-val 500 --n-test 500
  # Each CSV row = one H&E tile → one 3ch mIF stack (Hoechst/CD3e/Pan-CK). Not 16× rows.
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
# Zenodo md5:fdc3188206ac68576b4195cd039d9061 (~118 GiB)
ORION_ZIP_EXPECTED_BYTES = 127_020_270_255


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


def sample_split_df(df: pd.DataFrame, n_want: int | None, seed: int) -> pd.DataFrame:
    """Stratified subsample by slide. n_want=None → keep all rows."""
    if n_want is None or n_want <= 0 or len(df) <= n_want:
        return df.reset_index(drop=True)
    slide_col = _slide_col(df)
    n_slides = df[slide_col].nunique()
    per_slide = max(1, n_want // n_slides)
    parts = []
    for _, group in df.groupby(slide_col, sort=True):
        k = min(per_slide, len(group))
        parts.append(group.sample(n=k, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n_want:
        out = out.sample(n=n_want, random_state=seed).sort_index()
    return out.reset_index(drop=True)


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
        "ORIONCRC_dataset_tile_20x",
        "orioncrc_dataset_tile_20x",
    ):
        cand = data_dir / name
        if (cand / "train_dataframe.csv").exists():
            return cand
    if (data_dir / "train_dataframe.csv").exists():
        return data_dir
    # Zip may unpack with an extra top-level folder — search shallowly.
    for csv_path in data_dir.rglob("train_dataframe.csv"):
        root = csv_path.parent
        if (root / "he").is_dir() and (root / "if").is_dir():
            return root
    raise FileNotFoundError(
        f"No ORION tile root under {data_dir}. Expected train_dataframe.csv + he/ + if/\n"
        f"Download (~118 GiB): {ZENODO_ORION_URL}\n"
        f"Then: unzip ORIONCRC_dataset_tile_20x.zip -d {data_dir}"
    )


def _zip_ok(zip_path: Path) -> bool:
    if not zip_path.is_file():
        return False
    size = zip_path.stat().st_size
    # Allow 1% tolerance; partial urllib downloads are usually much smaller.
    if size < ORION_ZIP_EXPECTED_BYTES * 0.99:
        print(
            f"  [warn] zip size {size / 1e9:.2f} GB — expected ~{ORION_ZIP_EXPECTED_BYTES / 1e9:.1f} GB. "
            "Download likely incomplete; delete zip and re-download.",
            file=sys.stderr,
        )
        return False
    return True


def download_orion_zip(data_dir: Path) -> Path:
    """Download only. Returns path to zip."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ORIONCRC_dataset_tile_20x.zip"
    if _zip_ok(zip_path):
        print(f"Zip already complete: {zip_path}")
        return zip_path
    if zip_path.exists():
        print(f"Removing incomplete zip ({zip_path.stat().st_size / 1e9:.2f} GB)")
        zip_path.unlink()
    print(f"Downloading (~118 GiB) → {zip_path}")
    print("Use aria2c on login/long job; urllib is very slow for this file.")
    try:
            subprocess.run(
                [
                    "aria2c", "--check-certificate=false",
                    "-x", "16", "-s", "16", "-k", "10M",
                    "--file-allocation=none", "--continue=true",
                    "-o", str(zip_path.name), "-d", str(data_dir),
                    ZENODO_ORION_URL,
                ],
                check=True,
            )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        import urllib.request
        print(f"aria2c failed ({exc}); falling back to urllib...", file=sys.stderr)
        urllib.request.urlretrieve(ZENODO_ORION_URL, zip_path)
    if not _zip_ok(zip_path):
        raise RuntimeError(f"Incomplete download at {zip_path}. Re-run after deleting the zip.")
    return zip_path


def extract_orion_zip(data_dir: Path) -> Path:
    zip_path = data_dir / "ORIONCRC_dataset_tile_20x.zip"
    if not _zip_ok(zip_path):
        raise FileNotFoundError(f"Missing or incomplete zip: {zip_path}")
    try:
        return find_orion_root(data_dir)
    except FileNotFoundError:
        pass
    print(f"Extracting {zip_path} (this takes a while)...")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(data_dir)], check=True)
    return find_orion_root(data_dir)


def download_orion(data_dir: Path) -> Path:
    download_orion_zip(data_dir)
    return extract_orion_zip(data_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Orion-Lite pix2pix dataroot from MIPHEI ORION tiles.")
    p.add_argument("--src", type=str, default=None, help="Path to ORIONCRC_dataset_tile_20x (with CSVs + he/ + if/)")
    p.add_argument("--dst", type=str, default="./datasets/orion_lite", help="Output dataroot (trainA/B, ...)")
    p.add_argument(
        "--n-train",
        type=int,
        default=1500,
        help="Train tile pairs (stratified by slide). Each pair = 1 H&E + 1 3ch mIF stack.",
    )
    p.add_argument(
        "--n-val",
        type=int,
        default=None,
        metavar="N",
        help="Val tile pairs (stratified). Default: all official val rows (~12k).",
    )
    p.add_argument(
        "--n-test",
        type=int,
        default=None,
        metavar="N",
        help="Test tile pairs (stratified). Default: all official test rows (~11k).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile-size", type=int, default=512, help="Resize tiles (MIPHEI native=256; HEMIT protocol=512)")
    p.add_argument("--symlink-he", action="store_true", help="Symlink H&E at 256² (labels still extracted)")
    p.add_argument("--download", action="store_true", help="Download + unzip Zenodo zip into --data-dir")
    p.add_argument("--download-only", action="store_true", help="Download zip only (~118 GiB); no prep")
    p.add_argument("--extract-only", action="store_true", help="Unzip existing zip in --data-dir; no prep")
    p.add_argument("--data-dir", type=str, default="./data/orion", help="Download/extract parent when --download")
    args = p.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()

    if args.download_only:
        download_orion_zip(data_dir)
        print(f"Done. Next: python scripts/prepare_orion_lite.py --extract-only --data-dir {data_dir}")
        return
    if args.extract_only:
        src_root = extract_orion_zip(data_dir)
        print(f"Extracted at {src_root}")
        print(f"Next: python scripts/prepare_orion_lite.py --src {src_root}")
        return
    if args.download:
        src_root = download_orion(data_dir)
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

    train_sub = sample_split_df(train_df, args.n_train, args.seed)
    val_sub = sample_split_df(val_df, args.n_val, args.seed + 1)
    test_sub = sample_split_df(test_df, args.n_test, args.seed + 2)

    for name, sub, full in (
        ("train", train_sub, train_df),
        ("val", val_sub, val_df),
        ("test", test_sub, test_df),
    ):
        manifest = dst_root / f"{name}_manifest.csv"
        sub.to_csv(manifest, index=False)
        req = {"train": args.n_train, "val": args.n_val, "test": args.n_test}[name]
        req_s = "all" if req is None else str(req)
        print(f"{name} subset: {len(sub)} / {len(full)} rows (requested {req_s}) → {manifest}")

    counts = {}
    for split, df in ("train", train_sub), ("val", val_sub), ("test", test_sub):
        counts[split] = write_split(
            df, src_root, dst_root, split, args.tile_size, symlink_he=args.symlink_he
        )

    meta = {
        "source": str(src_root),
        "markers": ORION_LITE_MARKERS,
        "marker_indices": MARKER_TO_IDX,
        "n_train_requested": args.n_train,
        "n_val_requested": args.n_val,
        "n_test_requested": args.n_test,
        "source_csv_rows": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "tile_size": args.tile_size,
        "seed": args.seed,
        "splits": counts,
        "test_count": counts["test"],
        "slide_split": "MIPHEI official (37 train slides / 2 val / 2 test)",
        "note": "One row = one tile pair; 3 markers are channels in trainB/valB/testB, not separate images.",
    }
    meta_path = dst_root / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"meta.json → {meta_path}")
    print(f"Done. {sum(counts.values())} pairs total. Use: DATAROOT={dst_root}")


if __name__ == "__main__":
    main()
