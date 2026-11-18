---
name: univariate-analysis
description: Analyze one feature at a time with type-aware summaries and plots to identify quality, shape, and modeling implications.
---

# Univariate Analysis

## Purpose and use cases

Inspect individual training features before transformation. Use to find constants, rare categories, skew, suspicious codes, and scale differences without using evaluation data to choose preprocessing.

## Inputs and outputs

Inputs: training frame, semantic schema, optional weights, privacy rules. Outputs: per-feature summaries/plots, anomaly list, and transformation candidates with evidence.

## Execution and decision logic

1. Separate numeric, categorical, datetime, boolean, and text fields by semantic type.
2. Numeric: report finite count, robust quantiles, dispersion, skew, zero mass, and histogram/ECDF.
3. Categorical: report cardinality, top/rare levels, entropy, and missing category; datetime: coverage and frequency; text: length/token counts only when safe.
4. Flag constant, near-constant, near-unique, heavy-tailed, and impossible values.
5. Recommend transforms only when compatible with model family and domain meaning. High skew alone does not require a log transform; high cardinality alone does not justify dropping.

## Python example

```python
import pandas as pd


def numeric_summary(series: pd.Series) -> dict[str, float]:
    """Summarize one numeric feature robustly."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    q = clean.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "count": float(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "skew": float(clean.skew()),
        **{f"q{int(k * 100):02d}": float(v) for k, v in q.items()},
    }
```

## Validation checks and best practices

Reconcile missing and non-missing counts, label sampling, cap plot categories, and ensure axes do not conceal tails. Never expose raw PII in charts. Save summaries with data version and configuration; distinguish data defects from legitimate rare behavior.
