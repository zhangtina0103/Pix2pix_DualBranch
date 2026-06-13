#!/usr/bin/env python3
"""
Build HNSCC pix2pix dataroot from TCIA HNSCC-mIF-mIHC tiles.

Default task (cross_modal): mIHC hematoxylin → mIF CD3/CD8/FoxP3/PanCK.
Output labels are 4-channel uint8 TIFFs (one grayscale marker per channel).

Usage:
  python scripts/prepare_hnscc.py --src ~/Downloads/hnscc --dst ./datasets/hnscc
  python scripts/prepare_hnscc.py --src ./data/hnscc --mode max
  python scripts/prepare_hnscc.py --src ./data/hnscc --mode all_prefixes
  python scripts/prepare_hnscc.py --zip ~/Downloads/hnscc.zip --dst ./datasets/hnscc
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HNSCC_MARKERS = ["CD3", "CD8", "FoxP3", "PanCK"]
MIF_MARKER_FILES = {"PanCK": "PanCK", "CD3": "CD3", "CD8": "CD8", "FoxP3": "FoxP3"}
MIHC_MARKER_FILES = {"PanCK": "PCK", "CD3": "CD3", "CD8": "CD8", "FoxP3": "FoxP3"}

PREFIX_CONFIG = [
    ("Segmentation-", "hematoxylin", MIF_MARKER_FILES),
    ("mIF_Data-", "hematoxylin", MIF_MARKER_FILES),
    ("mIHC_Data-", "Hematoxylin", MIHC_MARKER_FILES),
]


def _resize_rgb(img: Image.Image, size: int) -> Image.Image:
    if img.size == (size, size):
        return img
    return img.resize((size, size), Image.Resampling.BILINEAR)


def _marker_gray(path: Path, tile_size: int) -> np.ndarray:
    return np.asarray(_resize_rgb(Image.open(path).convert("RGB"), tile_size).convert("L"), dtype=np.uint8)


def _case_splits(cases: list[str]) -> dict[str, list[str]]:
    n = len(cases)
    return {
        "train": cases[: int(n * 0.7)],
        "val": cases[int(n * 0.7) : int(n * 0.85)],
        "test": cases[int(n * 0.85) :],
    }


def _safe_stem(stem: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", stem)


def collect_cross_modal(root: Path, cases: list[str]) -> list[dict]:
    samples = []
    for case in cases:
        case_dir = root / case
        if not case_dir.is_dir():
            continue
        for he_path in sorted(case_dir.glob("mIHC_Data-*_Hematoxylin.png")):
            patch_key = he_path.stem.replace("mIHC_Data-", "").replace("_Hematoxylin", "")
            marker_paths = {
                m: case_dir / f"mIF_Data-{patch_key}_{MIF_MARKER_FILES[m]}.png" for m in HNSCC_MARKERS
            }
            if all(p.exists() for p in marker_paths.values()):
                samples.append(
                    {
                        "input_path": he_path,
                        "marker_paths": marker_paths,
                        "stem": _safe_stem(f"{case}_mIHC2mIF_{patch_key}"),
                        "mode": "cross_modal",
                    }
                )
    return samples


def collect_all_prefixes(root: Path, cases: list[str]) -> list[dict]:
    samples = []
    for case in cases:
        case_dir = root / case
        if not case_dir.is_dir():
            continue
        for prefix, he_suffix, marker_map in PREFIX_CONFIG:
            for he_path in sorted(case_dir.glob(f"{prefix}*_{he_suffix}.png")):
                patch_key = he_path.stem.replace(prefix, "").replace(f"_{he_suffix}", "")
                marker_paths = {}
                ok = True
                for marker in HNSCC_MARKERS:
                    file_marker = marker_map[marker]
                    mp = case_dir / f"{prefix}{patch_key}_{file_marker}.png"
                    if mp.exists():
                        marker_paths[marker] = mp
                    else:
                        ok = False
                        break
                if ok:
                    samples.append(
                        {
                            "input_path": he_path,
                            "marker_paths": marker_paths,
                            "stem": _safe_stem(f"{case}_{prefix.rstrip('-')}_{patch_key}"),
                            "mode": "all_prefixes",
                        }
                    )
    return samples


def collect_max(root: Path, cases: list[str]) -> list[dict]:
    """Union cross-modal (mIHC he → mIF GT) + all same-prefix groups. Dedupe by stem."""
    seen: set[str] = set()
    out: list[dict] = []
    for sample in collect_cross_modal(root, cases) + collect_all_prefixes(root, cases):
        if sample["stem"] in seen:
            continue
        seen.add(sample["stem"])
        sample = dict(sample)
        sample["mode"] = "max"
        out.append(sample)
    return out


COLLECTORS = {
    "cross_modal": collect_cross_modal,
    "all_prefixes": collect_all_prefixes,
    "max": collect_max,
}


def write_split(samples: list[dict], dst_root: Path, split: str, tile_size: int) -> int:
    dir_a = dst_root / f"{split}A"
    dir_b = dst_root / f"{split}B"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)
    n = 0
    for sample in samples:
        he = _resize_rgb(Image.open(sample["input_path"]).convert("RGB"), tile_size)
        out_he = dir_a / f"{sample['stem']}.tif"
        he.save(out_he)

        stack = np.stack(
            [_marker_gray(sample["marker_paths"][m], tile_size) for m in HNSCC_MARKERS],
            axis=-1,
        )
        planes = [Image.fromarray(stack[..., c], mode="L") for c in range(stack.shape[-1])]
        if len(planes) == 4:
            Image.merge("RGBA", planes).save(dir_b / f"{sample['stem']}.tif")
        else:
            Image.merge("RGB", planes[:3]).save(dir_b / f"{sample['stem']}.tif")
        n += 1
    print(f"  {split}: {n} pairs → {dir_a.name}/ , {dir_b.name}/")
    return n


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    for name in ("hnscc", "HNSCC"):
        cand = out_dir / name
        if cand.is_dir() and any(cand.iterdir()):
            return cand
    if any((out_dir / f"Case{i}").is_dir() for i in range(1, 9)):
        return out_dir
    raise FileNotFoundError(f"No HNSCC root found after extracting {zip_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare HNSCC pix2pix dataroot.")
    p.add_argument("--src", type=str, default=None, help="Path to extracted hnscc/ (Case1..Case8)")
    p.add_argument("--zip", type=str, default=None, help="Path to hnscc.zip (extract then prepare)")
    p.add_argument("--dst", type=str, default="./datasets/hnscc", help="Output dataroot")
    p.add_argument(
        "--mode",
        choices=tuple(COLLECTORS),
        default="max",
        help="max: union cross_modal + all_prefixes (most data); cross_modal: mIHC→mIF only",
    )
    p.add_argument("--tile-size", type=int, default=512)
    args = p.parse_args()

    if args.zip:
        zip_path = Path(args.zip).expanduser().resolve()
        extract_parent = Path(args.dst).expanduser().resolve().parent / "raw_hnscc"
        src_root = extract_zip(zip_path, extract_parent)
    elif args.src:
        src_root = Path(args.src).expanduser().resolve()
    else:
        p.error("Provide --src or --zip")

    if not src_root.is_dir():
        raise FileNotFoundError(f"HNSCC root not found: {src_root}")

    dst_root = Path(args.dst).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    cases = sorted(d.name for d in src_root.iterdir() if d.is_dir() and d.name.startswith("Case"))
    if not cases:
        raise FileNotFoundError(f"No Case* folders under {src_root}")

    split_cases = _case_splits(cases)
    collector = COLLECTORS[args.mode]

    print(f"HNSCC src: {src_root}")
    print(f"Mode: {args.mode}")
    print(f"Cases: {cases}")
    print(f"Markers: {HNSCC_MARKERS} (output_nc=4)")
    print(f"pix2pix dst: {dst_root}")

    counts = {}
    manifests = {}
    for split, case_list in split_cases.items():
        samples = collector(src_root, case_list)
        counts[split] = write_split(samples, dst_root, split, args.tile_size)
        manifest = dst_root / f"{split}_manifest.txt"
        manifest.write_text("\n".join(s["stem"] for s in samples) + ("\n" if samples else ""))
        manifests[split] = len(samples)

    meta = {
        "source": str(src_root),
        "mode": args.mode,
        "markers": HNSCC_MARKERS,
        "output_nc": len(HNSCC_MARKERS),
        "tile_size": args.tile_size,
        "case_split": split_cases,
        "splits": counts,
        "test_count": counts["test"],
        "note": (
            "max: cross_modal (mIHC he→mIF GT) + Segmentation/mIF/mIHC same-prefix groups; "
            "trainA=hematoxylin RGB; trainB=4ch marker stack (CD3,CD8,FoxP3,PanCK)"
        ),
    }
    meta_path = dst_root / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"meta.json → {meta_path}")
    print(f"Done. {sum(counts.values())} pairs total. DATAROOT={dst_root}")


if __name__ == "__main__":
    main()
