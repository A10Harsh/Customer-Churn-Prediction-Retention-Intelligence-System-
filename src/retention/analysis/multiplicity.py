"""Validated multiple-testing corrections with explicit underflow handling."""

import numpy as np
import pandas as pd

SMALLEST_POSITIVE_FLOAT = np.nextafter(0.0, 1.0)


def adjustment_ready(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Validate p-values and replace underflowed zeros by a documented bound."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ValueError("p-values must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be finite and within [0, 1].")
    return np.where(values == 0.0, SMALLEST_POSITIVE_FLOAT, values)


def benjamini_hochberg(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted p-values in original order."""
    values = adjustment_ready(p_values)
    count = len(values)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * count / np.arange(1, count + 1))[::-1]
    )[::-1]
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def holm(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Return Holm FWER-adjusted p-values in original order."""
    values = adjustment_ready(p_values)
    count = len(values)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate(ranked * np.arange(count, 0, -1))
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def adjust_hypothesis_families(
    tests: pd.DataFrame,
    alpha: float,
    fdr_threshold: float,
) -> pd.DataFrame:
    """Apply BH and Holm separately within each declared hypothesis family."""
    required = {"hypothesis_id", "hypothesis_family", "raw_p_value"}
    missing = required - set(tests.columns)
    if missing:
        raise ValueError(f"Missing correction columns: {sorted(missing)}")
    if not 0 < alpha < 1 or not 0 < fdr_threshold < 1:
        raise ValueError("alpha and fdr_threshold must be between zero and one.")
    parts = []
    for _, family in tests.groupby("hypothesis_family", sort=False):
        family = family.copy()
        raw = family["raw_p_value"].to_numpy(dtype=float)
        family["p_value_used_for_adjustment"] = adjustment_ready(raw)
        family["p_value_underflow_bounded"] = raw == 0.0
        family["bh_adjusted_p_value"] = benjamini_hochberg(raw)
        family["holm_adjusted_p_value"] = holm(raw)
        family["raw_reject"] = raw <= alpha
        family["bh_reject"] = family["bh_adjusted_p_value"] <= fdr_threshold
        family["holm_reject"] = family["holm_adjusted_p_value"] <= alpha
        parts.append(family)
    return pd.concat(parts, ignore_index=True)
