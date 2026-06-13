#!/usr/bin/env python3
"""Build HEMIT robustness manifest from paper_models.csv + existing test TIFFs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_image_dir(srcdir: Path) -> Path:
    """Inline helper so manifest build works even if hemit.eval import fails."""
    images = srcdir / "images"
    if images.is_dir() and any(p.name.endswith("_fake_B.tif") for p in images.iterdir()):
        return images
    if any(p.name.endswith("_fake_B.tif") for p in srcdir.iterdir()):
        return srcdir
    raise FileNotFoundError(f"No *_fake_B.tif under {srcdir} or {srcdir}/images")


def _has_images(srcdir: Path) -> bool:
    try:
        image_dir = resolve_image_dir(srcdir)
    except FileNotFoundError:
        return False
    return any(image_dir.glob("*_fake_B.tif"))


def _has_checkpoint(repo: Path, train_name: str, epoch: int) -> bool:
    ckpt = repo / "checkpoints" / train_name / f"{epoch}_net_G.pth"
    ckpt_a = repo / "checkpoints" / train_name / f"{epoch}_net_G_A.pth"
    return ckpt.is_file() or ckpt_a.is_file()


def load_paper_models(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("model", "").startswith("#"):
                continue
            if not row.get("model") or not row.get("train_name"):
                continue
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Build eval/hemit/manifest.csv from paper models.")
    p.add_argument("--paper-models", type=str, default="eval/hemit/paper_models.csv")
    p.add_argument("--out", type=str, default="eval/hemit/manifest.csv")
    p.add_argument("--test-epoch", type=int, default=None,
                   help="Override epoch for all models (default: per-row test_epoch)")
    p.add_argument("--repo", type=str, default=".")
    p.add_argument("--require-images", action="store_true",
                   help="Only include models with existing test TIFFs")
    p.add_argument("--status", action="store_true",
                   help="Print checkpoint/image status and exit")
    args = p.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    paper_path = (repo / args.paper_models).resolve()
    models = load_paper_models(paper_path)

    status_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []

    for m in models:
        epoch = args.test_epoch if args.test_epoch is not None else int(m.get("test_epoch") or 80)
        train_name = m["train_name"]
        srcdir = repo / "results" / train_name / f"test_{epoch}" / "images"
        has_ckpt = _has_checkpoint(repo, train_name, epoch)
        has_img = _has_images(srcdir)
        status_rows.append({
            "model": m["model"],
            "train_name": train_name,
            "epoch": str(epoch),
            "checkpoint": "yes" if has_ckpt else "no",
            "test_images": "yes" if has_img else "no",
            "srcdir": str(srcdir),
        })
        if has_img or not args.require_images:
            if args.require_images and not has_img:
                continue
            if has_img:
                manifest_rows.append({"model": m["model"], "srcdir": str(srcdir)})

    print(f"paper_models: {paper_path}")
    print(f"{'model':<12} {'epoch':>5} {'ckpt':>5} {'tiff':>5}  train_name")
    print("-" * 72)
    for s in status_rows:
        print(f"{s['model']:<12} {s['epoch']:>5} {s['checkpoint']:>5} {s['test_images']:>5}  {s['train_name']}")

    if args.status:
        missing = [s for s in status_rows if s["test_images"] == "no"]
        if missing:
            print(f"\n{len(missing)} model(s) need test.py first (no TIFFs).")
            print("Run: sbatch bash_scripts/submit_hemit_full_robustness.sbatch")
        return

    out_path = (repo / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "srcdir"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    ready = sum(1 for s in status_rows if s["test_images"] == "yes")
    print(f"\nWrote {out_path} ({len(manifest_rows)} rows, {ready}/{len(status_rows)} have TIFFs)")
    if ready < len(status_rows):
        print("Some models missing TIFFs — run submit_hemit_full_robustness.sbatch to generate them.")


if __name__ == "__main__":
    main()
