#!/usr/bin/env python3
"""
Audit CaMSC intensity normalization after harmonize + k-fold prep.

Checks:
  1. Per-field stats (BF / Hoechst / WT1) on the flat harmonized pool
  2. Old vs new batch balance (index <= ref_max_index = old, passthrough)
  3. Optional: prepared fold trainA/trainB match the source pool (tile_size=0)
  4. Optional: before/after vs raw src (--compare-src)

Usage (on cluster, after prepare_camsc_bf.py):
  python scripts/audit_camsc_normalization.py \\
    --src ~/orcd/scratch/camsc/camsc_all_harm \\
    --kfold-root ~/orcd/scratch/camsc/datasets/camsc_bf_kfold_125_harm \\
    --compare-src ~/orcd/scratch/camsc/camsc_all \\
    --ref-max-index 10 \\
    --out ~/orcd/scratch/camsc/audit_kfold_125_harm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

_FILENAME_RE = re.compile(
    r"^CaMSC\s+(\d+)%\s+(\d+x)\s+(BF|Hoechst|WT1)(?:_+(\d+))?\.tif$",
    re.IGNORECASE,
)


def _load_gray(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = np.max(arr[..., :3], axis=-1)
        elif arr.shape[0] in (3, 4):
            arr = np.max(arr[:3], axis=0)
        else:
            arr = arr[..., 0]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() <= 1.0:
        arr *= 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _load_label_stack(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = tifffile.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D label at {path}, got {arr.shape}")
    if arr.shape[-1] >= 2:
        hoechst = np.asarray(arr[..., 0], dtype=np.float32)
        wt1 = np.asarray(arr[..., 1], dtype=np.float32)
    else:
        raise ValueError(f"Unexpected label shape {arr.shape} at {path}")
    if hoechst.max() <= 1.0:
        hoechst *= 255.0
        wt1 *= 255.0
    return (
        np.clip(hoechst, 0, 255).astype(np.uint8),
        np.clip(wt1, 0, 255).astype(np.uint8),
    )


def _channel_stats(arr: np.ndarray, wt1_nonzero: bool, marker: str) -> dict[str, float]:
    flat = arr.ravel()
    if wt1_nonzero and marker.upper() == "WT1":
        nz = flat[flat > 0]
        vals = nz if nz.size else flat
    else:
        vals = flat
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p01": float(np.percentile(vals, 1)),
        "p50": float(np.percentile(vals, 50)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(arr.max()),
        "frac_sat": float((arr >= 254).mean()),
    }


def _resolve(src: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return src / p


def _batch_name(index: int, ref_max_index: int) -> str:
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return "unknown"
    return "old" if idx <= ref_max_index else "new"


def audit_manifest_pool(
    src: Path,
    manifest: Path,
    ref_max_index: int,
    wt1_nonzero: bool,
) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    rows: list[dict] = []
    for _, row in df.iterrows():
        sample_id = str(row.get("sample_id", ""))
        idx = int(row["index"]) if "index" in row and pd.notna(row["index"]) else -1
        pct = str(row.get("pct", ""))
        batch = _batch_name(idx, ref_max_index)

        paths = {
            "BF": _resolve(src, str(row["bf_path"])),
            "Hoechst": _resolve(src, str(row["hoechst_path"])),
            "WT1": _resolve(src, str(row["wt1_path"])),
        }
        entry: dict = {
            "sample_id": sample_id,
            "pct": pct,
            "index": idx,
            "batch": batch,
        }
        for marker, path in paths.items():
            if not path.is_file():
                entry[f"{marker}_missing"] = True
                continue
            stats = _channel_stats(_load_gray(path), wt1_nonzero, marker)
            for k, v in stats.items():
                entry[f"{marker}_{k}"] = v
        rows.append(entry)
    return pd.DataFrame(rows)


def _sample_id_from_stem(stem: str) -> str | None:
    m = re.search(r"(camsc_\d+pct_\d+x_\d+)", stem, re.I)
    return m.group(1).lower() if m else None


def audit_raw_vs_harm(
    manifest: Path,
    raw_src: Path,
    harm_src: Path,
    wt1_nonzero: bool,
) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    rows = []
    for _, row in df.iterrows():
        sid = str(row.get("sample_id", ""))
        idx = int(row["index"])
        for marker, col in (
            ("BF", "bf_path"),
            ("Hoechst", "hoechst_path"),
            ("WT1", "wt1_path"),
        ):
            raw_p = _resolve(raw_src, str(row[col]))
            harm_p = _resolve(harm_src, str(row[col]))
            if not raw_p.is_file() or not harm_p.is_file():
                continue
            raw = _load_gray(raw_p)
            harm = _load_gray(harm_p)
            rows.append({
                "sample_id": sid,
                "index": idx,
                "marker": marker,
                "raw_mean": float(raw.mean()),
                "harm_mean": float(harm.mean()),
                "delta_mean": float(harm.mean() - raw.mean()),
                "raw_p99": float(np.percentile(raw, 99)),
                "harm_p99": float(np.percentile(harm, 99)),
            })
    return pd.DataFrame(rows)


def audit_prepared_fold(
    fold_dir: Path,
    pool_df: pd.DataFrame,
    atol: float = 0.5,
) -> pd.DataFrame:
    """Verify trainA/B (and val/test) match pool stats when tile_size=0."""
    by_id = pool_df.copy()
    by_id["_sid"] = by_id["sample_id"].astype(str).str.lower()
    by_id = by_id.set_index("_sid", drop=False)
    rows = []
    for split in ("train", "val", "test"):
        dir_a = fold_dir / f"{split}A"
        dir_b = fold_dir / f"{split}B"
        if not dir_a.is_dir():
            continue
        for bf_path in sorted(dir_a.glob("*.tif")):
            stem = bf_path.stem
            sid_guess = _sample_id_from_stem(stem) or stem

            label_path = dir_b / bf_path.name
            if not label_path.is_file():
                rows.append({
                    "split": split,
                    "file": bf_path.name,
                    "status": "missing_label",
                })
                continue

            bf = _load_gray(bf_path)
            hoechst, wt1 = _load_label_stack(label_path)

            pool_row = None
            if sid_guess in by_id.index:
                pool_row = by_id.loc[sid_guess]
            else:
                for sid, prow in by_id.iterrows():
                    if str(sid).lower() == sid_guess.lower():
                        pool_row = prow
                        sid_guess = str(sid)
                        break

            row = {
                "split": split,
                "file": bf_path.name,
                "sample_id": sid_guess,
                "prep_BF_mean": float(bf.mean()),
                "prep_Hoechst_mean": float(hoechst.mean()),
                "prep_WT1_mean": float(wt1.mean()),
            }
            if pool_row is not None:
                for marker, prep_mean in (
                    ("BF", row["prep_BF_mean"]),
                    ("Hoechst", row["prep_Hoechst_mean"]),
                    ("WT1", row["prep_WT1_mean"]),
                ):
                    pool_mean = pool_row.get(f"{marker}_mean", np.nan)
                    row[f"pool_{marker}_mean"] = pool_mean
                    row[f"delta_{marker}"] = (
                        prep_mean - pool_mean if pd.notna(pool_mean) else np.nan
                    )
                max_delta = max(
                    abs(row.get("delta_BF", 0) or 0),
                    abs(row.get("delta_Hoechst", 0) or 0),
                    abs(row.get("delta_WT1", 0) or 0),
                )
                row["status"] = "ok" if max_delta <= atol else "mismatch"
            else:
                row["status"] = "no_pool_match"
            rows.append(row)
    return pd.DataFrame(rows)


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    flags = []
    for _, row in df.iterrows():
        issues = []
        for marker in ("BF", "Hoechst", "WT1"):
            mean = row.get(f"{marker}_mean", np.nan)
            p99 = row.get(f"{marker}_p99", np.nan)
            sat = row.get(f"{marker}_frac_sat", np.nan)
            if pd.isna(mean):
                issues.append(f"{marker}_missing")
                continue
            if mean < 5:
                issues.append(f"{marker}_very_dark")
            if mean > 220:
                issues.append(f"{marker}_very_bright")
            if p99 >= 254 and row.get("batch") == "new":
                issues.append(f"{marker}_p99_saturated")
            if sat > 0.05:
                issues.append(f"{marker}_high_sat_frac")
        # Cross-marker imbalance (new batch only)
        if row.get("batch") == "new":
            bf = row.get("BF_mean", np.nan)
            wt1 = row.get("WT1_mean", np.nan)
            if pd.notna(bf) and pd.notna(wt1) and abs(bf - wt1) > 120:
                issues.append("BF_WT1_mean_gap")
        flags.append(";".join(issues) if issues else "")
    out = df.copy()
    out["flags"] = flags
    return out


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = []
    for marker in ("BF", "Hoechst", "WT1"):
        col = f"{marker}_mean"
        if col not in df.columns:
            continue
        g = df.groupby(group_cols, dropna=False)[col].agg(["count", "mean", "std", "min", "max"])
        g.columns = pd.MultiIndex.from_product([[marker], g.columns])
        metrics.append(g)
    if not metrics:
        return pd.DataFrame()
    return pd.concat(metrics, axis=1)


def _plot_box_means(df: pd.DataFrame, out: Path) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, marker in zip(axes, ("BF", "Hoechst", "WT1")):
        col = f"{marker}_mean"
        if col not in df.columns:
            continue
        batches = []
        for batch in ("old", "new"):
            vals = df.loc[df["batch"] == batch, col].dropna().values
            if vals.size:
                batches.append(vals)
            else:
                batches.append(np.array([]))
        ax.boxplot(batches, tick_labels=["old", "new"])
        ax.set_title(marker)
        ax.set_ylabel("mean intensity")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-field mean intensity: old (passthrough) vs new (harmonized)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_scatter_bf_wt1(df: pd.DataFrame, out: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    for batch, color in (("old", "C0"), ("new", "C1")):
        sub = df[df["batch"] == batch]
        ax.scatter(
            sub["BF_mean"],
            sub["WT1_mean"],
            c=color,
            label=batch,
            alpha=0.75,
            s=40,
        )
    ax.set_xlabel("BF mean")
    ax.set_ylabel("WT1 mean")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("BF vs WT1 mean per field")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_by_pct(df: pd.DataFrame, out: Path) -> None:
    if plt is None:
        return
    pcts = sorted(df["pct"].astype(str).unique())
    markers = ("BF", "Hoechst", "WT1")
    fig, axes = plt.subplots(len(markers), 1, figsize=(10, 8), sharex=True)
    if len(markers) == 1:
        axes = [axes]
    x = np.arange(len(pcts))
    width = 0.35
    for ax, marker in zip(axes, markers):
        col = f"{marker}_mean"
        old_means = []
        new_means = []
        for pct in pcts:
            sub = df[df["pct"].astype(str) == pct]
            old_means.append(sub.loc[sub["batch"] == "old", col].mean())
            new_means.append(sub.loc[sub["batch"] == "new", col].mean())
        ax.bar(x - width / 2, old_means, width, label="old", alpha=0.85)
        ax.bar(x + width / 2, new_means, width, label="new", alpha=0.85)
        ax.set_ylabel(f"{marker} mean")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{p}%" for p in pcts])
    axes[-1].set_xlabel("O2 group")
    fig.suptitle("Mean intensity by O2% and batch")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Audit CaMSC normalization after k-fold prep")
    p.add_argument("--src", type=str, required=True, help="Harmonized flat TIF pool")
    p.add_argument(
        "--kfold-root",
        type=str,
        default="",
        help="Prepared k-fold root (manifest_all.csv + fold0..)",
    )
    p.add_argument("--fold", type=int, default=0, help="Fold to verify against pool")
    p.add_argument("--compare-src", type=str, default="", help="Raw pool before harmonize")
    p.add_argument("--ref-max-index", type=int, default=10)
    p.add_argument("--wt1-nonzero", action="store_true", default=True)
    p.add_argument("--prep-atol", type=float, default=0.5,
                   help="Max |prep_mean - pool_mean| for tile_size=0 match")
    p.add_argument("--out", type=str, default="", help="Output directory for CSV/plots")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Missing --src: {src}")

    kfold_root = Path(args.kfold_root).expanduser().resolve() if args.kfold_root else None
    manifest = (kfold_root / "manifest_all.csv") if kfold_root else src / "manifest_all.csv"
    if kfold_root and (kfold_root / "manifest_all.csv").is_file():
        manifest = kfold_root / "manifest_all.csv"
    elif not manifest.is_file():
        raise SystemExit(
            f"Manifest not found: {manifest}\n"
            "Pass --kfold-root with manifest_all.csv from prepare_camsc_bf.py"
        )

    out_dir = Path(args.out).expanduser().resolve() if args.out else (
        kfold_root / "audit_norm" if kfold_root else src / "audit_norm"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pool src:   {src}")
    print(f"Manifest:   {manifest}")
    print(f"Ref index:  <= {args.ref_max_index} = old batch\n")

    pool_df = audit_manifest_pool(src, manifest, args.ref_max_index, args.wt1_nonzero)
    pool_df = flag_outliers(pool_df)
    pool_csv = out_dir / "per_field_stats.csv"
    pool_df.to_csv(pool_csv, index=False)
    print(f"Fields: {len(pool_df)}  → {pool_csv}")

    n_flagged = int((pool_df["flags"] != "").sum())
    print(f"Flagged fields: {n_flagged}")
    if n_flagged:
        flagged = pool_df[pool_df["flags"] != ""][
            ["sample_id", "pct", "index", "batch", "BF_mean", "Hoechst_mean", "WT1_mean", "flags"]
        ]
        print(flagged.to_string(index=False))
        flagged.to_csv(out_dir / "flagged_fields.csv", index=False)

    batch_summary = summarize(pool_df, ["batch"])
    pct_summary = summarize(pool_df, ["pct", "batch"])
    batch_summary.to_csv(out_dir / "summary_by_batch.csv")
    pct_summary.to_csv(out_dir / "summary_by_pct_batch.csv")
    print("\nMean intensity by batch:")
    print(batch_summary.to_string())

    if args.compare_src:
        raw_src = Path(args.compare_src).expanduser().resolve()
        cmp_df = audit_raw_vs_harm(manifest, raw_src, src, args.wt1_nonzero)
        cmp_df.to_csv(out_dir / "raw_vs_harm.csv", index=False)
        # Old batch should be ~0 delta
        old_ids = set(pool_df.loc[pool_df["batch"] == "old", "sample_id"])
        old_cmp = cmp_df[cmp_df["sample_id"].isin(old_ids)]
        max_old_delta = old_cmp["delta_mean"].abs().max() if not old_cmp.empty else 0.0
        print(f"\nRaw vs harm: max |Δmean| on old batch = {max_old_delta:.3f} (expect ~0)")

    if kfold_root and (kfold_root / f"fold{args.fold}").is_dir():
        fold_dir = kfold_root / f"fold{args.fold}"
        prep_df = audit_prepared_fold(fold_dir, pool_df, atol=args.prep_atol)
        prep_df.to_csv(out_dir / f"fold{args.fold}_prep_check.csv", index=False)
        n_mismatch = int((prep_df["status"] == "mismatch").sum())
        n_ok = int((prep_df["status"] == "ok").sum())
        print(f"\nFold {args.fold} prep check: {n_ok} ok, {n_mismatch} mismatch")
        if n_mismatch:
            bad = prep_df[prep_df["status"] == "mismatch"].head(10)
            print(bad.to_string(index=False))

        counts = {}
        for split in ("train", "val", "test"):
            for side in ("A", "B"):
                d = fold_dir / f"{split}{side}"
                counts[f"{split}{side}"] = len(list(d.glob("*.tif"))) if d.is_dir() else 0
        print("Prepared file counts:", counts)

    if plt is not None:
        _plot_box_means(pool_df, out_dir / "box_mean_by_batch.png")
        _plot_scatter_bf_wt1(pool_df, out_dir / "scatter_bf_vs_wt1.png")
        _plot_by_pct(pool_df, out_dir / "bar_mean_by_pct.png")
        print(f"\nPlots → {out_dir}/*.png")

    meta = {
        "src": str(src),
        "manifest": str(manifest),
        "kfold_root": str(kfold_root) if kfold_root else None,
        "ref_max_index": args.ref_max_index,
        "n_fields": len(pool_df),
        "n_flagged": n_flagged,
        "out_dir": str(out_dir),
    }
    (out_dir / "audit_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
