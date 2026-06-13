#!/usr/bin/env python3
"""Build HEMIT robustness manifest from existing test TIFFs in results/."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Friendly short names for paper tables (fallback: strip hemit_ prefix)
MODEL_ALIASES: dict[str, str] = {
    "hemit_fm_cross_attn_scratch": "cross_attn",
    "hemit_pix2pix_resnet9": "pix2pix",
    "hemit_cut_joint": "cut",
    "hemit_asp_joint": "asp",
    "hemit_cyclegan_joint": "cyclegan",
    "hemit_vanilla_fm_joint_perc": "vanilla_fm",
    "hemit_vanilla_fm_joint_perc_scratch": "vanilla_fm_scratch",
    "hemit_SwinTResnet_New_2": "dualbranch",
}


def resolve_image_dir(srcdir: Path) -> Path:
    images = srcdir / "images"
    if images.is_dir() and any(p.name.endswith("_fake_B.tif") for p in images.iterdir()):
        return images
    if any(p.name.endswith("_fake_B.tif") for p in srcdir.iterdir()):
        return srcdir
    raise FileNotFoundError(f"No *_fake_B.tif under {srcdir} or {srcdir}/images")


def _has_images(srcdir: Path) -> bool:
    try:
        return any(resolve_image_dir(srcdir).glob("*_fake_B.tif"))
    except FileNotFoundError:
        return False


def _has_checkpoint(repo: Path, train_name: str, epoch: int) -> bool:
    ckpt = repo / "checkpoints" / train_name / f"{epoch}_net_G.pth"
    ckpt_a = repo / "checkpoints" / train_name / f"{epoch}_net_G_A.pth"
    return ckpt.is_file() or ckpt_a.is_file()


def _model_label(train_name: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    base = train_name.removesuffix("_512") if train_name.endswith("_512") else train_name
    return MODEL_ALIASES.get(base, MODEL_ALIASES.get(train_name, train_name.removeprefix("hemit_") or train_name))


def _is_hemit_model(train_name: str) -> bool:
    return train_name.startswith("hemit_")


def load_paper_models(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = (row.get("model") or "").strip()
            train_name = (row.get("train_name") or "").strip()
            if not model or not train_name or model.startswith("#"):
                continue
            rows.append(row)
    return rows


def discover_from_results(repo: Path, epoch: int) -> list[dict[str, str]]:
    """Scan results/<train_name>/test_<epoch>/images for TIFF pairs."""
    results_root = repo / "results"
    if not results_root.is_dir():
        return []
    discovered: list[dict[str, str]] = []
    for train_dir in sorted(results_root.iterdir()):
        if not train_dir.is_dir():
            continue
        train_name = train_dir.name
        srcdir = train_dir / f"test_{epoch}" / "images"
        if not _has_images(srcdir):
            continue
        discovered.append({
            "model": _model_label(train_name),
            "train_name": train_name,
            "test_epoch": str(epoch),
            "srcdir": str(srcdir.resolve()),
        })
    return discovered


def build_model_list(
    repo: Path,
    *,
    paper_models: list[dict[str, str]],
    epoch: int,
    auto_discover: bool,
) -> list[dict[str, str]]:
    """Merge paper_models.csv with auto-discovery from results/."""
    by_train: dict[str, dict[str, str]] = {}

    if paper_models:
        for m in paper_models:
            ep = epoch if epoch is not None else int(m.get("test_epoch") or 80)
            train_name = m["train_name"]
            srcdir = repo / "results" / train_name / f"test_{ep}" / "images"
            by_train[train_name] = {
                "model": _model_label(train_name, m.get("model")),
                "train_name": train_name,
                "test_epoch": str(ep),
                "srcdir": str(srcdir.resolve()),
            }

    if auto_discover or not paper_models:
        for d in discover_from_results(repo, epoch):
            if d["train_name"] not in by_train:
                by_train[d["train_name"]] = d
            else:
                # refresh srcdir from disk
                by_train[d["train_name"]]["srcdir"] = d["srcdir"]

    return list(by_train.values())


def main() -> None:
    p = argparse.ArgumentParser(description="Build eval/hemit/manifest.csv from results/ TIFFs.")
    p.add_argument("--paper-models", type=str, default="eval/hemit/paper_models.csv",
                   help="Optional; auto-discovers from results/ if missing")
    p.add_argument("--out", type=str, default="eval/hemit/manifest.csv")
    p.add_argument("--test-epoch", type=int, default=80)
    p.add_argument("--repo", type=str, default=".")
    p.add_argument("--require-images", action="store_true",
                   help="Only include models with existing test TIFFs")
    p.add_argument("--include-orion", action="store_true",
                   help="Include orion_lite_* runs (default: HEMIT hemit_* only)")
    p.add_argument("--no-auto-discover", action="store_true",
                   help="Only use paper_models.csv (error if missing)")
    p.add_argument("--status", action="store_true", help="Print status table and exit")
    args = p.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    paper_path = (repo / args.paper_models).resolve()
    paper_models = load_paper_models(paper_path) if paper_path.is_file() else []

    if args.no_auto_discover and not paper_models:
        raise SystemExit(f"ERROR: missing {paper_path} and --no-auto-discover set")

    models = build_model_list(
        repo,
        paper_models=paper_models,
        epoch=args.test_epoch,
        auto_discover=not args.no_auto_discover,
    )

    if not models:
        raise SystemExit(
            f"No models found under {repo / 'results'} for test_{args.test_epoch}/images.\n"
            f"Check --test-epoch or run test.py first."
        )

    if not args.include_orion:
        models = [m for m in models if _is_hemit_model(m["train_name"])]

    if paper_path.is_file():
        print(f"paper_models: {paper_path}")
    else:
        print(f"paper_models: (not found — auto-discovered from {repo / 'results'})")

    status_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []

    for m in models:
        train_name = m["train_name"]
        epoch = int(m.get("test_epoch") or args.test_epoch)
        srcdir = Path(m["srcdir"])
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
        if has_img:
            manifest_rows.append({"model": m["model"], "srcdir": str(srcdir)})
        elif not args.require_images:
            manifest_rows.append({"model": m["model"], "srcdir": str(srcdir)})

    if args.require_images:
        manifest_rows = [r for r in manifest_rows
                         if _has_images(Path(r["srcdir"]))]

    print(f"{'model':<16} {'epoch':>5} {'ckpt':>5} {'tiff':>5}  train_name")
    print("-" * 76)
    for s in sorted(status_rows, key=lambda x: x["model"]):
        print(f"{s['model']:<16} {s['epoch']:>5} {s['checkpoint']:>5} {s['test_images']:>5}  {s['train_name']}")

    if args.status:
        missing = [s for s in status_rows if s["test_images"] == "no"]
        if missing:
            print(f"\n{len(missing)} model(s) missing TIFFs @ epoch {args.test_epoch}")
        return

    if not manifest_rows:
        raise SystemExit("No models with test TIFFs — nothing to write to manifest.")

    out_path = (repo / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "srcdir"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote {out_path} ({len(manifest_rows)} models)")
    print("Next:")
    print("  python scripts/run_hemit_robustness_eval.py \\")
    print("    --manifest eval/hemit/manifest.csv \\")
    print("    --outdir eval/hemit/robustness_comparison \\")
    print("    --reference-model pix2pix")


if __name__ == "__main__":
    main()
