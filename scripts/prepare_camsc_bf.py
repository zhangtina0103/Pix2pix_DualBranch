#!/usr/bin/env python3
"""
Build pix2pix dataroot for CaMSC brightfield → Hoechst + WT1 virtual staining.

Pairs BF + Hoechst + WT1 by naming convention:
  CaMSC {pct}% {mag} BF_{n}.tif  ↔  Hoechst_{n}.tif  ↔  WT1_{n}.tif
(10% BF uses double underscore: BF__8.tif ↔ Hoechst_8.tif)

Output (same layout as HEMIT / Orion-Lite):
  datasets/camsc_bf/
    trainA/ trainB/  valA/ valB/  testA/ testB/
    meta.json
    manifest_used.csv

Usage:
  # Auto-discover triplets from 20260504 folder (recommended)
  python scripts/prepare_camsc_bf.py \\
    --src ~/Downloads/20260504 --auto-discover --dst ./datasets/camsc_bf

  # Stratified 5-fold CV (fold0..fold4 under dst)
  python scripts/prepare_camsc_bf.py \\
    --src ~/orcd/scratch/camsc/20260504 --auto-discover \\
    --k-folds 5 --dst ./datasets/camsc_bf_kfold

  # Aggregate test metrics after all folds trained:
  python scripts/eval_camsc_kfold.py --kfold-root ./datasets/camsc_bf_kfold \\
    --results-root ./results --name-prefix camsc_bf_vanilla_fm_fold
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

MARKERS = ("Hoechst", "WT1")

_FILENAME_RE = re.compile(
    r"^CaMSC\s+(\d+)%\s+(\d+x)\s+(BF|Hoechst|WT1)(?:_+(\d+))?\.tif$",
    re.IGNORECASE,
)


def _resolve(src: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return src / p


def _load_gray(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        # RGB or multi-channel — take max projection or first channel
        if arr.shape[-1] in (3, 4):
            arr = np.max(arr[..., :3], axis=-1)
        elif arr.shape[0] in (3, 4):
            arr = np.max(arr[:3], axis=0)
        else:
            arr = arr[..., 0]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _resize_gray(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    img = Image.fromarray(arr)
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _resize_stack(stack: np.ndarray, size: int) -> np.ndarray:
    if stack.shape[0] == size and stack.shape[1] == size:
        return stack
    out = np.zeros((size, size, stack.shape[2]), dtype=stack.dtype)
    for c in range(stack.shape[2]):
        out[..., c] = _resize_gray(stack[..., c], size)
    return out


def _gray_to_rgb(arr: np.ndarray) -> np.ndarray:
    return np.stack([arr, arr, arr], axis=-1)


def discover_triplets(src: Path, mag: str = "10x") -> list[dict]:
    """Find BF+Hoechst+WT1 triplets under src by filename convention."""
    groups: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for path in sorted(src.glob("*.tif")):
        m = _FILENAME_RE.match(path.name)
        if not m:
            continue
        pct, file_mag, kind, num = m.group(1), m.group(2), m.group(3), m.group(4)
        if file_mag != mag:
            continue
        if not num:
            continue  # skip e.g. CaMSC 30% 4x BF.tif (no fluorescence pair)
        key = (pct, file_mag, num)
        groups[key][kind.upper() if kind.upper() == "BF" else kind] = path.name

    rows = []
    for (pct, file_mag, num), kinds in sorted(groups.items()):
        if not all(k in kinds for k in ("BF", "Hoechst", "WT1")):
            continue
        rows.append({
            "pct": pct,
            "mag": file_mag,
            "index": num,
            "bf_path": kinds["BF"],
            "hoechst_path": kinds["Hoechst"],
            "wt1_path": kinds["WT1"],
            "sample_id": f"camsc_{pct}pct_{file_mag}_{num}",
        })
    return rows


def assign_splits(
    rows: list[dict],
    seed: int = 42,
    n_val_per_group: int = 1,
    n_test_per_group: int = 1,
) -> pd.DataFrame:
    """Stratified train/val/test by O2 concentration (pct)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    rng = np.random.default_rng(seed)
    splits = []
    for pct, group in df.groupby("pct", sort=True):
        idxs = group.index.to_list()
        rng.shuffle(idxs)
        n_val = min(n_val_per_group, max(0, len(idxs) - 1))
        n_test = min(n_test_per_group, max(0, len(idxs) - n_val - 1))
        val_set = set(idxs[:n_val])
        test_set = set(idxs[n_val:n_val + n_test])
        for i in idxs:
            if i in val_set:
                splits.append("val")
            elif i in test_set:
                splits.append("test")
            else:
                splits.append("train")
    df = df.copy()
    df["split"] = splits
    return df


def assign_kfold_ids(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Assign fold 0..k-1 within each O2 (pct) group (stratified)."""
    out = df.copy()
    out["fold"] = -1
    rng = np.random.default_rng(seed)
    for _pct, group in out.groupby("pct", sort=True):
        idxs = group.index.to_list()
        rng.shuffle(idxs)
        if len(idxs) % k != 0:
            raise ValueError(
                f"pct={_pct}: {len(idxs)} samples not divisible by k={k}"
            )
        per_fold = len(idxs) // k
        for i, idx in enumerate(idxs):
            out.loc[idx, "fold"] = i // per_fold
    return out


def split_for_fold(
    df: pd.DataFrame,
    fold: int,
    seed: int,
    n_val_per_group: int = 1,
) -> pd.DataFrame:
    """Build train/val/test for one CV fold (test = held-out fold)."""
    if "fold" not in df.columns:
        raise ValueError("DataFrame missing fold column — run assign_kfold_ids first")

    out = df.copy()
    splits: dict[int, str] = {}
    rng = np.random.default_rng(seed + fold)

    for _pct, group in out.groupby("pct", sort=True):
        idxs = group.index.to_list()
        test_idxs = {i for i in idxs if int(out.loc[i, "fold"]) == fold}
        pool = [i for i in idxs if i not in test_idxs]
        rng.shuffle(pool)
        n_val = min(n_val_per_group, max(0, len(pool) - 1))
        val_idxs = set(pool[:n_val])
        for i in idxs:
            if i in test_idxs:
                splits[i] = "test"
            elif i in val_idxs:
                splits[i] = "val"
            else:
                splits[i] = "train"

    out["split"] = [splits[i] for i in out.index]
    return out


def build_dataroot(
    df: pd.DataFrame,
    src: Path,
    dst: Path,
    tile_size: int,
) -> tuple[int, pd.DataFrame]:
    """Write trainA/B, valA/B, testA/B. Returns (n_written, used_df)."""
    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    used_rows = []
    for split in ("train", "val", "test"):
        sub = df[df["split"].astype(str).str.lower() == split]
        if sub.empty:
            print(f"  [skip] no rows for split={split}")
            continue
        total += write_split(sub, src, dst, split, tile_size)
        used_rows.append(sub)
    if total == 0:
        raise SystemExit("No pairs written. Check manifest paths and split column.")
    used = pd.concat(used_rows, ignore_index=True)
    used.to_csv(dst / "manifest_used.csv", index=False)
    return total, used


def write_meta(
    dst: Path,
    *,
    total: int,
    tile_size: int,
    src: Path,
    manifest: Path,
    k_folds: int | None = None,
    fold: int | None = None,
) -> None:
    meta = {
        "task": "brightfield_to_hoechst_wt1",
        "markers": list(MARKERS),
        "label_channels": ["Hoechst", "WT1", "pad"],
        "train_output_nc": 3,
        "fm_channel_weights": "1,1,0",
        "n_pairs": total,
        "tile_size": tile_size,
        "src": str(src),
        "manifest": str(manifest),
    }
    if k_folds is not None:
        meta["k_folds"] = k_folds
    if fold is not None:
        meta["fold"] = fold
    (dst / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _safe_id(row: pd.Series, idx: int) -> str:
    if pd.notna(row.get("sample_id")) and str(row["sample_id"]).strip():
        stem = re.sub(r"[^\w.\-]+", "_", str(row["sample_id"]).strip())
        return stem
    bf = Path(str(row["bf_path"])).stem
    bf = re.sub(r"[^\w.\-]+", "_", bf)
    return f"{idx:04d}_{bf}"


def write_split(
    df: pd.DataFrame,
    src: Path,
    dst: Path,
    split: str,
    tile_size: int,
) -> int:
    phase = split
    dir_a = dst / f"{phase}A"
    dir_b = dst / f"{phase}B"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    n = 0
    for idx, row in df.iterrows():
        bf_path = _resolve(src, str(row["bf_path"]))
        hoechst_path = _resolve(src, str(row["hoechst_path"]))
        wt1_path = _resolve(src, str(row["wt1_path"]))

        missing = [p for p in (bf_path, hoechst_path, wt1_path) if not p.exists()]
        if missing:
            print(f"  [warn] skip row {idx}: missing {missing[0]}", file=sys.stderr)
            continue

        stem = _safe_id(row, idx)
        out_bf = dir_a / f"{stem}.tif"
        out_label = dir_b / f"{stem}.tif"

        bf = _resize_gray(_load_gray(bf_path), tile_size)
        hoechst = _resize_gray(_load_gray(hoechst_path), tile_size)
        wt1 = _resize_gray(_load_gray(wt1_path), tile_size)

        # 3ch label: [Hoechst, WT1, pad] — train with output_nc=3, FM_CHANNEL_WEIGHTS=1,1,0
        label = np.stack([hoechst, wt1, np.zeros_like(hoechst)], axis=-1)
        tifffile.imwrite(out_bf, _gray_to_rgb(bf))
        tifffile.imwrite(out_label, label)
        n += 1

    print(f"  {split}: {n} pairs → {dir_a.name}/ , {dir_b.name}/")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare CaMSC brightfield → Hoechst+WT1 dataroot")
    p.add_argument("--src", type=str, required=True, help="Folder with CaMSC *.tif files (e.g. 20260504)")
    p.add_argument("--manifest", type=str, default="", help="Optional CSV manifest")
    p.add_argument("--auto-discover", action="store_true",
                   help="Pair BF/Hoechst/WT1 by filename (CaMSC {pct}% {mag} *_N.tif)")
    p.add_argument("--mag", type=str, default="10x", help="Magnification filter for --auto-discover")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-val-per-group", type=int, default=1,
                   help="Val images per O2 concentration group")
    p.add_argument("--n-test-per-group", type=int, default=1,
                   help="Test images per O2 concentration group")
    p.add_argument("--k-folds", type=int, default=0,
                   help="If >0, write stratified K-fold dataroots to dst/fold0..fold{K-1}")
    p.add_argument("--dst", type=str, default="./datasets/camsc_bf")
    p.add_argument("--tile-size", type=int, default=512)
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.is_dir():
        raise SystemExit(f"Source not found: {src}")

    if args.auto_discover or not args.manifest:
        rows = discover_triplets(src, mag=args.mag)
        if not rows:
            raise SystemExit(f"No complete triplets found in {src} (mag={args.mag})")
        base_df = pd.DataFrame(rows)
        manifest = dst / "manifest_all.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        base_df.to_csv(manifest, index=False)
        print(f"Auto-discovered {len(base_df)} triplets → {manifest}")
    else:
        manifest = Path(args.manifest).expanduser().resolve()
        if not manifest.is_file():
            raise SystemExit(f"Manifest not found: {manifest}")
        base_df = pd.read_csv(manifest)
        required = {"bf_path", "hoechst_path", "wt1_path"}
        missing_cols = required - set(base_df.columns)
        if missing_cols:
            raise SystemExit(f"Manifest missing columns: {sorted(missing_cols)}")

    print(f"CaMSC src: {src}")
    print(f"manifest:  {manifest}")
    print(f"markers:   {MARKERS} (+ 1 padded channel for loader)")

    if args.k_folds > 0:
        k = args.k_folds
        print(f"K-fold:    k={k} stratified by O2% → {dst}/fold0..fold{k - 1}")
        df_k = assign_kfold_ids(base_df, k=k, seed=args.seed)
        fold_summaries = []
        grand_total = 0
        for fold in range(k):
            fold_df = split_for_fold(
                df_k, fold, seed=args.seed, n_val_per_group=args.n_val_per_group,
            )
            fold_dst = dst / f"fold{fold}"
            print(f"\n==> fold {fold} → {fold_dst}")
            print(fold_df.groupby("split").size().to_string())
            n, _ = build_dataroot(fold_df, src, fold_dst, args.tile_size)
            write_meta(
                fold_dst, total=n, tile_size=args.tile_size, src=src,
                manifest=manifest, k_folds=k, fold=fold,
            )
            fold_summaries.append({
                "fold": fold,
                "train": int((fold_df["split"] == "train").sum()),
                "val": int((fold_df["split"] == "val").sum()),
                "test": int((fold_df["split"] == "test").sum()),
            })
            grand_total += n

        folds_json = {
            "k_folds": k,
            "seed": args.seed,
            "n_val_per_group": args.n_val_per_group,
            "src": str(src),
            "folds": fold_summaries,
        }
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "folds.json").write_text(json.dumps(folds_json, indent=2) + "\n")
        df_k.to_csv(dst / "manifest_kfold.csv", index=False)
        print(f"\nDone. {grand_total} image writes across {k} folds.")
        print(f"folds.json → {dst / 'folds.json'}")
        print(
            f"Train: CAMSC_KFOLD_ROOT={dst} sbatch bash_scripts/train_camsc_bf_kfold_array.sbatch"
        )
        return

    # Single train/val/test split
    if "split" not in base_df.columns:
        df = assign_splits(
            base_df.to_dict("records"),
            seed=args.seed,
            n_val_per_group=args.n_val_per_group,
            n_test_per_group=args.n_test_per_group,
        )
        manifest_auto = dst / "manifest_auto.csv"
        manifest_auto.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(manifest_auto, index=False)
        manifest = manifest_auto
        print(df.groupby("split").size().to_string())
    else:
        df = base_df

    print(f"pix2pix dst: {dst}")
    n, _ = build_dataroot(df, src, dst, args.tile_size)
    write_meta(dst, total=n, tile_size=args.tile_size, src=src, manifest=manifest)
    print(f"Done. {n} pairs. manifest_used → {dst / 'manifest_used.csv'}")
    print(f"Train: DATAROOT={dst} INPUT_NC=3 OUTPUT_NC=3 FM_CHANNEL_WEIGHTS=1,1,0")


if __name__ == "__main__":
    main()
