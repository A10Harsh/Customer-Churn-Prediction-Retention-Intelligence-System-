---
name: distribution-analysis
description: Diagnose feature shape, tails, mixtures, zero inflation, truncation, and subgroup or temporal distribution changes.
---

# Distribution Analysis

## Purpose and use cases

Decide whether features need robust treatment, transforms, special indicators, or investigation. Use on training data and for explicitly labeled reference-versus-current comparisons.

## Inputs and outputs

Inputs: feature, semantic bounds, optional groups/time/reference, weights, plotting budget. Outputs: quantiles, shape diagnostics, plots, distribution comparison metrics, and justified treatment candidates.

## Execution and decision logic

1. Separate missing, zero, finite, and infinite values.
2. Plot histogram plus ECDF/QQ or categorical frequency; show raw and sensible transformed scale when needed.
3. Inspect truncation, heaping, multimodality, floor/ceiling effects, and subgroup/time shifts.
4. Select robust scaling, log1p, power transform, bins, or a two-part indicator only when domain and model benefit support it.
5. Never infer a population distribution from convenience samples without documenting sampling.

## Python example

```python
import numpy as np
import pandas as pd


def distribution_facts(series: pd.Series) -> dict[str, float]:
    """Return tail and mass diagnostics for a numeric series."""
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    return {
        "missing_rate": float(values.isna().mean()),
        "zero_rate": float((finite == 0).mean()),
        "skew": float(finite.skew()),
        "q01": float(finite.quantile(0.01)),
        "q99": float(finite.quantile(0.99)),
    }
```

## Validation checks and best practices

Reconcile all masses, retain axis units, report sample sizes, and ensure transformations are defined for zeros/negatives. Fit distribution transforms on training folds only. Prefer robust summaries over fragile fitted-family labels and verify any treatment by downstream CV.
