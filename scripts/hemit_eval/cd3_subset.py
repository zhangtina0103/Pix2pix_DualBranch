"""CD3 tile subset definitions: nucleus-positive vs mean-intensity enrichment."""

from __future__ import annotations

import numpy as np

CD3_CHANNEL_INDEX = 1


def tile_mean_cd3_intensity(stack: np.ndarray) -> float:
    """Mean CD3 channel intensity on a multiplex stack (H, W, 3)."""
    return float(np.mean(stack[..., CD3_CHANNEL_INDEX]))


def cd3_enrichment_threshold(mean_cd3_scores: np.ndarray, top_frac: float) -> float:
    """
    Score cutoff for the top ``top_frac`` fraction of tiles by mean real CD3.

    Example: top_frac=0.10 → tiles with mean CD3 >= 90th percentile.
    """
    scores = np.asarray(mean_cd3_scores, dtype=np.float64)
    if scores.size == 0:
        return float("nan")
    if not 0.0 < top_frac < 1.0:
        raise ValueError(f"top_frac must be in (0, 1), got {top_frac}")
    return float(np.percentile(scores, (1.0 - top_frac) * 100.0))


def is_cd3_enriched_tile(mean_cd3: float, threshold: float) -> bool:
    return bool(np.isfinite(mean_cd3) and np.isfinite(threshold) and mean_cd3 >= threshold)
