#!/usr/bin/env python3
"""Build eval/diffusion/manifest.csv from diffusion baseline TIFF exports."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# model label → relative srcdir under repo (or absolute in paper_models.csv)
DEFAULT_DIFFUSION_DIRS: dict[str, str] = {
    "dvst": "results/diffusion/dvst/images",
    "dvst_zero_shot": "results/diffusion/dvst/images",
    "diffvs": "results/diffusion/diffvs/images",
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


def _expand_path(repo: Path, raw: str) -> Path:
    expanded = os.path.expandvars(raw)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = (repo / path).resolve()
    return path.resolve()


def load_paper_models(path: Path, repo: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = (row.get("model") or "").strip()
            srcdir = (row.get("srcdir") or "").strip()
            if not model or not srcdir or model.startswith("#"):
                continue
            rows.append({
                "model": model,
                "srcdir": str(_expand_path(repo, srcdir)),
                "family": (row.get("family") or "diffusion").strip(),
                "checkpoint": (row.get("checkpoint") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            })
    return rows


def discover_under_diffusion_root(repo: Path) -> list[dict[str, str]]:
    """Scan results/diffusion/<name>/images for TIFF pairs."""
    root = repo / "results" / "diffusion"
    if not root.is_dir():
        return []
    found: list[dict[str, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        srcdir = child / "images"
        if not _has_images(srcdir):
            continue
        label = child.name
        found.append({
            "model": label,
            "srcdir": str(resolve_image_dir(srcdir)),
            "family": "diffusion",
            "checkpoint": "",
            "notes": f"auto: results/diffusion/{label}/images",
        })
    return found


def discover_diffvs_fallback(repo: Path) -> list[dict[str, str]]:
    """DiffVS exports to <inference_dir>/pix2pix_metrics by default."""
    candidates: list[str] = []
    infer = os.environ.get("DIFFVS_INFER_DIR", "").strip()
    if infer:
        candidates.append(str(Path(infer) / "pix2pix_metrics"))
    diffvs_root = os.environ.get("DIFFVS_ROOT", str(repo.parent / "DiffVS")).strip()
    candidates.append(str(Path(diffvs_root) / "outputs/hemit512/inference/pix2pix_metrics"))
    candidates.append(str(repo / "results/diffusion/diffvs/images"))

    seen: set[str] = set()
    found: list[dict[str, str]] = []
    for raw in candidates:
        srcdir = _expand_path(repo, raw)
        key = str(srcdir)
        if key in seen or not _has_images(srcdir):
            continue
        seen.add(key)
        found.append({
            "model": "diffvs",
            "srcdir": str(resolve_image_dir(srcdir)),
            "family": "diffusion",
            "checkpoint": "",
            "notes": f"auto: {srcdir}",
        })
    return found


def build_entries(
    repo: Path,
    *,
    paper_models: list[dict[str, str]],
    auto_discover: bool,
    include_diffvs: bool,
) -> list[dict[str, str]]:
    by_model: dict[str, dict[str, str]] = {}

    for m in paper_models:
        by_model[m["model"]] = m

    if auto_discover:
        for d in discover_under_diffusion_root(repo):
            if d["model"] not in by_model:
                by_model[d["model"]] = d

        for label, rel in DEFAULT_DIFFUSION_DIRS.items():
            if label in by_model:
                continue
            srcdir = _expand_path(repo, rel)
            if _has_images(srcdir):
                by_model[label] = {
                    "model": label,
                    "srcdir": str(resolve_image_dir(srcdir)),
                    "family": "diffusion",
                    "checkpoint": "",
                    "notes": f"default: {rel}",
                }

        if include_diffvs and "diffvs" not in by_model:
            for d in discover_diffvs_fallback(repo):
                by_model.setdefault(d["model"], d)

    return list(by_model.values())


def main() -> None:
    p = argparse.ArgumentParser(description="Build eval/diffusion/manifest.csv for robustness eval.")
    p.add_argument("--paper-models", type=str, default="eval/diffusion/paper_models.csv",
                   help="Optional explicit model list (model,srcdir,...)")
    p.add_argument("--out", type=str, default="eval/diffusion/manifest.csv")
    p.add_argument("--repo", type=str, default=".")
    p.add_argument("--require-images", action="store_true",
                   help="Only include models with existing TIFF pairs")
    p.add_argument("--no-auto-discover", action="store_true",
                   help="Only use paper_models.csv")
    p.add_argument("--no-diffvs", action="store_true",
                   help="Skip DiffVS fallback paths")
    p.add_argument("--status", action="store_true", help="Print table and exit")
    args = p.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    paper_path = (repo / args.paper_models).resolve()
    paper_models = load_paper_models(paper_path, repo) if paper_path.is_file() else []

    if args.no_auto_discover and not paper_models:
        raise SystemExit(f"ERROR: missing {paper_path} and --no-auto-discover set")

    entries = build_entries(
        repo,
        paper_models=paper_models,
        auto_discover=not args.no_auto_discover,
        include_diffvs=not args.no_diffvs,
    )

    if paper_path.is_file():
        print(f"paper_models: {paper_path}")
    else:
        print(f"paper_models: (not found — auto-discovering under {repo / 'results/diffusion'})")

    status_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []

    for e in entries:
        srcdir = Path(e["srcdir"])
        has_img = _has_images(srcdir)
        status_rows.append({
            "model": e["model"],
            "family": e.get("family", "diffusion"),
            "tiff": "yes" if has_img else "no",
            "srcdir": str(srcdir),
            "checkpoint": e.get("checkpoint", ""),
            "notes": e.get("notes", ""),
        })
        if has_img or not args.require_images:
            manifest_rows.append({"model": e["model"], "srcdir": str(srcdir)})

    if args.require_images:
        manifest_rows = [r for r in manifest_rows if _has_images(Path(r["srcdir"]))]

    print(f"{'model':<20} {'family':<10} {'tiff':>5}  srcdir")
    print("-" * 90)
    for s in sorted(status_rows, key=lambda x: x["model"]):
        print(f"{s['model']:<20} {s['family']:<10} {s['tiff']:>5}  {s['srcdir']}")

    if args.status:
        missing = [s for s in status_rows if s["tiff"] == "no"]
        if missing:
            print(f"\n{len(missing)} diffusion model(s) missing TIFFs")
        return

    if not manifest_rows:
        raise SystemExit(
            "No diffusion TIFFs found.\n"
            "Run D-VST export first:\n"
            "  MODEL=dvst MODE=test|metrics bash scripts/run_hemit_all.sh\n"
            "Or add rows to eval/diffusion/paper_models.csv"
        )

    out_path = (repo / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "srcdir"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote {out_path} ({len(manifest_rows)} models)")
    print("Next:")
    print("  python scripts/run_hemit_robustness_eval.py \\")
    print("    --manifest eval/diffusion/manifest.csv \\")
    print("    --outdir eval/diffusion/robustness_comparison \\")
    print("    --reference-model pix2pix")
    print("\nOr merge with HEMIT GAN/FM:")
    print("  python scripts/concat_robustness_manifests.py \\")
    print("    --manifest eval/hemit/manifest.csv \\")
    print("    --manifest eval/diffusion/manifest.csv \\")
    print("    --out eval/combined/manifest.csv")


if __name__ == "__main__":
    main()
