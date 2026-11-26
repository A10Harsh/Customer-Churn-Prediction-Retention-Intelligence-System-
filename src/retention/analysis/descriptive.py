"""Pure descriptive statistics used by the automated EDA report."""

from typing import Any

import numpy as np
import pandas as pd


def dataset_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Return dimensions and simple data-quality counts."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "feature_cardinality": {c: int(df[c].nunique(dropna=False)) for c in df},
    }


def numerical_statistics(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Calculate notebook-aligned sample moments and robust summaries."""
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        q1, median, q3 = values.quantile([0.25, 0.50, 0.75])
        rows.append(
            {
                "feature": feature,
                "count": int(values.count()),
                "missing": int(values.isna().sum()),
                "mean": values.mean(),
                "median": median,
                "standard_deviation": values.std(ddof=1),
                "variance": values.var(ddof=1),
                "minimum": values.min(),
                "q1": q1,
                "q3": q3,
                "maximum": values.max(),
                "iqr": q3 - q1,
                "skewness": values.skew(),
                "excess_kurtosis": values.kurt(),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def categorical_frequencies(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return count and percentage for every categorical level."""
    rows: list[dict[str, Any]] = []
    for feature in features:
        counts = df[feature].value_counts(dropna=False)
        for value, count in counts.items():
            rows.append(
                {
                    "feature": feature,
                    "category": "<missing>" if pd.isna(value) else str(value),
                    "count": int(count),
                    "percentage": float(count / len(df) * 100) if len(df) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def rare_categories(frequencies: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Screen categories whose population share is strictly below ``threshold``."""
    result = frequencies.copy()
    result["rare"] = result["percentage"].lt(threshold * 100)
    return result[result["rare"]].reset_index(drop=True)


def variability_screen(
    df: pd.DataFrame, features: list[str], dominance_threshold: float
) -> pd.DataFrame:
    """Identify constant and dominant-value near-constant features."""
    rows = []
    for feature in features:
        proportions = df[feature].value_counts(dropna=False, normalize=True)
        dominance = float(proportions.iloc[0]) if not proportions.empty else np.nan
        rows.append(
            {
                "feature": feature,
                "unique_values": int(len(proportions)),
                "dominant_percentage": dominance * 100,
                "constant": bool(len(proportions) <= 1),
                "near_constant": bool(dominance >= dominance_threshold),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def target_summary(
    df: pd.DataFrame, target: str, positive: str = "Yes"
) -> dict[str, Any]:
    """Summarize target balance and the majority-class classification baseline."""
    counts = df[target].value_counts(dropna=False)
    if len(counts) != 2 or positive not in counts.index:
        raise ValueError(
            "Target must contain exactly two classes including the positive label."
        )
    majority = counts.idxmax()
    minority = counts.idxmin()
    total = int(counts.sum())
    majority_count = int(counts.loc[majority])
    minority_count = int(counts.loc[minority])
    positive_count = int(counts.loc[positive])
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "proportions": {str(k): float(v / total) for k, v in counts.items()},
        "positive_label": positive,
        "positive_count": positive_count,
        "positive_rate": positive_count / total,
        "majority_label": str(majority),
        "minority_label": str(minority),
        "imbalance_ratio": majority_count / minority_count,
        "majority_baseline": {
            "accuracy": majority_count / total,
            "positive_precision": 0.0
            if majority != positive
            else majority_count / total,
            "positive_recall": 0.0 if majority != positive else 1.0,
            "positive_f1": 0.0
            if majority != positive
            else 2 * (majority_count / total) / (1 + majority_count / total),
            "specificity": 1.0 if majority != positive else 0.0,
            "balanced_accuracy": 0.5,
        },
    }
