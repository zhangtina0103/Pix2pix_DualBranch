"""Bootstrap CIs and summary stats for per-tile HEMIT metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.stats import bootstrap as _scipy_bootstrap
except ImportError:  # pragma: no cover
    _scipy_bootstrap = None


def summarize_values(
    values: np.ndarray,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    random_state: int = 42,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"), "higher_is_better": higher_is_better,
        }
    ci_low, ci_high = compute_bootstrap_ci(
        arr, confidence_level=confidence_level, n_resamples=n_resamples, random_state=random_state,
    )
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "higher_is_better": higher_is_better,
    }


def compute_bootstrap_ci(
    values: np.ndarray,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    random_state: int | None = 42,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if np.all(values == values[0]):
        return float(values[0]), float(values[0])
    if _scipy_bootstrap is not None:
        actual_resamples = min(n_resamples, max(100, values.size * 10))
        try:
            result = _scipy_bootstrap(
                (values,), statistic=np.mean, n_resamples=actual_resamples,
                confidence_level=confidence_level, method="BCa", random_state=random_state,
            )
            return float(result.confidence_interval.low), float(result.confidence_interval.high)
        except Exception:
            result = _scipy_bootstrap(
                (values,), statistic=np.mean, n_resamples=actual_resamples,
                confidence_level=confidence_level, method="percentile", random_state=random_state,
            )
            return float(result.confidence_interval.low), float(result.confidence_interval.high)
    return _percentile_bootstrap_ci(values, confidence_level, n_resamples, random_state)


def _percentile_bootstrap_ci(
    values: np.ndarray, confidence_level: float, n_resamples: int, random_state: int | None,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n = values.size
    means = [float(np.mean(values[rng.integers(0, n, size=n)])) for _ in range(min(n_resamples, max(500, n * 20)))]
    alpha = 1.0 - confidence_level
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def compare_paired(
    baseline: np.ndarray, variant: np.ndarray, *, metric_name: str,
    n_resamples: int = 10000, random_state: int = 42,
) -> dict[str, Any]:
    b = np.asarray(baseline, dtype=np.float64).ravel()
    v = np.asarray(variant, dtype=np.float64).ravel()
    if b.size != v.size:
        raise ValueError(f"Paired arrays differ in length: {b.size} vs {v.size}")
    diff = v - b
    ci_low, ci_high = compute_bootstrap_ci(diff, n_resamples=n_resamples, random_state=random_state)
    return {
        "metric": metric_name,
        "baseline_mean": float(np.nanmean(b)),
        "variant_mean": float(np.nanmean(v)),
        "mean_diff": float(np.nanmean(diff)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": bool((ci_low > 0) or (ci_high < 0)),
    }
