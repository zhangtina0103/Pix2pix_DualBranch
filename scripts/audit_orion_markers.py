#!/usr/bin/env python3
"""
Rank ORION mIF markers by sparsity (FM-friendly) vs continuity (GAN-friendly).

Sparse / punctate markers → flow matching + Pearson tends to win vs GANs on HEMIT-like tasks.
Continuous epithelial / stromal / vascular stains → GAN texture matching tends to win.

Usage (cluster, ~5–15 min on 200 tiles):
  python scripts/audit_orion_markers.py \\
    --src ~/orcd/scratch/orion/ORIONCRC_dataset_tile_20x \\
    --n-tiles 200 --seed 42

Then prepare a 3-marker dataroot:
  python scripts/prepare_orion_lite.py --src ... \\
    --dst ~/orcd/scratch/orion/datasets/orion_immune_cd3_cd8_foxp3 \\
    --markers CD3e,CD8a,FOXP3 --n-train 1480 --n-val 500 --n-test 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Keep in sync with prepare_orion_lite.py
ORION_CHANNEL_ORDER = [
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO",
    "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA",
]

# Heuristic tags (literature + Orion pan-CK post-mortem)
GAN_FRIENDLY = {"Pan-CK", "E-cadherin", "SMA", "CD31", "Hoechst"}
IMMUNE_SPARSE = {
    "CD3e", "CD8a", "CD4", "FOXP3", "PD-1", "CD20", "CD45RO", "PD-L1", "CD45",
}
MACROPHAGE_PATCHY = {"CD68", "CD163"}


def _resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute() and p.exists():
        return p
    if (root / p).exists():
        return root / p
    name = p.name
    for sub in ("he", "if", "nuclei"):
        cand = root / sub / name
        if cand.exists():
            return cand
    return root / p


def find_orion_root(data_dir: Path) -> Path:
    if (data_dir / "train_dataframe.csv").exists():
        return data_dir
    for csv_path in data_dir.rglob("train_dataframe.csv"):
        root = csv_path.parent
        if (root / "if").is_dir():
            return root
    raise FileNotFoundError(f"No ORION root under {data_dir}")


def load_full_stack(if_path: Path) -> np.ndarray:
    arr = tifffile.imread(if_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D mIF at {if_path}, got {arr.shape}")
    if arr.shape[-1] >= len(ORION_CHANNEL_ORDER):
        return np.asarray(arr[..., : len(ORION_CHANNEL_ORDER)], dtype=np.float32)
    if arr.shape[0] >= len(ORION_CHANNEL_ORDER):
        return np.asarray(arr[: len(ORION_CHANNEL_ORDER)], dtype=np.float32).transpose(1, 2, 0)
    raise ValueError(f"Unexpected shape {arr.shape} at {if_path}")


def channel_stats(ch: np.ndarray, thresh_pct: float = 95.0) -> dict:
    """thresh = percentile of nonzero pixels used as 'positive' cutoff."""
    flat = ch.ravel()
    nz = flat[flat > 0]
    if nz.size == 0:
        return {"pos_frac": 0.0, "mean": 0.0, "p95": 0.0}
    thr = float(np.percentile(nz, thresh_pct))
    pos_frac = float((ch >= thr).mean())
    return {"pos_frac": pos_frac, "mean": float(ch.mean()), "p95": thr}


def fm_score(pos_frac: float, name: str) -> float:
    """Higher = more FM-favorable (sparser immune, not epithelial/vascular)."""
    s = 1.0 - min(pos_frac * 8.0, 1.0)  # sparsity reward
    if name in GAN_FRIENDLY:
        s -= 0.35
    if name in IMMUNE_SPARSE:
        s += 0.15
    if name in MACROPHAGE_PATCHY:
        s -= 0.05
    return s


def suggest_panels(ranked: list[tuple[str, float, float]]) -> list[dict]:
    """Return a few 3-marker panels: immune-only + one with Hoechst."""
    by_name = {n: (sc, pf) for n, sc, pf in ranked}
    immune = [n for n, sc, _ in ranked if n in IMMUNE_SPARSE and n not in GAN_FRIENDLY]
    panels = []

    def add_panel(name: str, markers: list[str], note: str) -> None:
        if len(markers) == 3 and all(m in by_name for m in markers):
            panels.append({
                "name": name,
                "markers": markers,
                "note": note,
                "avg_fm_score": float(np.mean([by_name[m][0] for m in markers])),
                "avg_pos_frac": float(np.mean([by_name[m][1] for m in markers])),
            })

    if len(immune) >= 3:
        top3 = immune[:3]
        add_panel(
            "immune_top3_sparse",
            top3,
            "Data-driven: 3 sparsest immune markers from audit",
        )
    add_panel(
        "immune_cd3_cd8_foxp3",
        ["CD3e", "CD8a", "FOXP3"],
        "Recommended default: T-cell panel, no nuclear/epithelial",
    )
    add_panel(
        "immune_cd3_foxp3_pd1",
        ["CD3e", "FOXP3", "PD-1"],
        "Ultra-sparse checkpoints (if audit ranks FOXP3/PD-1 high)",
    )
    add_panel(
        "hoechst_cd3_cd4",
        ["Hoechst", "CD3e", "CD4"],
        "Recommended screen: keep HEMIT core, drop Pan-CK for moderate immune CD4",
    )
    add_panel(
        "hemit_matched",
        ["Hoechst", "CD3e", "Pan-CK"],
        "Current panel (GAN-favored on CRC; use as supplement only)",
    )
    # dedupe by marker set
    seen = set()
    out = []
    for p in sorted(panels, key=lambda x: -x["avg_fm_score"]):
        key = tuple(p["markers"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Audit ORION marker sparsity for FM vs GAN panel choice.")
    p.add_argument("--src", type=str, required=True, help="ORIONCRC_dataset_tile_20x root")
    p.add_argument("--n-tiles", type=int, default=200, help="Train tiles to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thresh-pct", type=float, default=95.0, help="Positive pixel percentile within channel")
    p.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON summary")
    args = p.parse_args()

    src = find_orion_root(Path(args.src).expanduser().resolve())
    df = pd.read_csv(src / "train_dataframe.csv")
    n = min(args.n_tiles, len(df))
    sub = df.sample(n=n, random_state=args.seed)

    accum: dict[str, list[float]] = {m: [] for m in ORION_CHANNEL_ORDER}
    used = 0
    for _, row in sub.iterrows():
        if_path = _resolve_path(src, str(row["target_path"]))
        if not if_path.exists():
            continue
        try:
            stack = load_full_stack(if_path)
        except (ValueError, OSError) as exc:
            print(f"  [skip] {if_path.name}: {exc}", file=sys.stderr)
            continue
        for i, name in enumerate(ORION_CHANNEL_ORDER):
            st = channel_stats(stack[..., i], thresh_pct=args.thresh_pct)
            accum[name].append(st["pos_frac"])
        used += 1

    if used == 0:
        print("ERROR: no tiles loaded", file=sys.stderr)
        sys.exit(1)

    print(f"ORION marker audit: {src}")
    print(f"Tiles sampled: {used} (requested {n})")
    print(f"Positive fraction = % pixels >= p{args.thresh_pct:.0f} within channel (lower → sparser)\n")
    print(f"{'Marker':<12} {'pos_frac':>9} {'FM score':>9}  tag")
    print("-" * 50)

    ranked: list[tuple[str, float, float]] = []
    for name in ORION_CHANNEL_ORDER:
        pf = float(np.mean(accum[name]))
        sc = fm_score(pf, name)
        ranked.append((name, sc, pf))
        tags = []
        if name in GAN_FRIENDLY:
            tags.append("GAN-friendly")
        if name in IMMUNE_SPARSE:
            tags.append("immune")
        print(f"{name:<12} {pf:>9.4f} {sc:>9.3f}  {','.join(tags)}")

    ranked.sort(key=lambda x: (-x[1], x[2]))
    print("\n=== Ranked for FM-favorability (higher FM score = sparser immune, not epithelial) ===")
    for i, (name, sc, pf) in enumerate(ranked[:10], 1):
        print(f"  {i:2d}. {name:<12} FM={sc:.3f}  pos_frac={pf:.4f}")

    panels = suggest_panels(ranked)
    print("\n=== Suggested 3-marker panels (prepare with --markers a,b,c) ===")
    for pnl in panels:
        m = ",".join(pnl["markers"])
        print(
            f"  [{pnl['name']}]  --markers {m}\n"
            f"      avg_fm_score={pnl['avg_fm_score']:.3f}  avg_pos_frac={pnl['avg_pos_frac']:.4f}\n"
            f"      {pnl['note']}"
        )

    best = panels[0]
    print("\n>>> Pick:", ",".join(best["markers"]), f"({best['name']})")
    print("    If immune_top3 overlaps cd3_cd8_foxp3, use cd3_cd8_foxp3 for cleaner paper text.")

    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        payload = {
            "src": str(src),
            "n_tiles": used,
            "ranked": [{"marker": n, "fm_score": sc, "pos_frac": pf} for n, sc, pf in ranked],
            "panels": panels,
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
