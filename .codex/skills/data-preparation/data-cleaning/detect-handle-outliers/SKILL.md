---
name: detect-handle-outliers
description: Distinguish invalid extremes from legitimate rare observations and apply train-fitted, model-aware outlier treatment.
---

# Detect and Handle Outliers

## Purpose and use cases

Identify values that are impossible, influential, or distributionally unusual. Use after type correction and before scaling/modeling.

## Inputs and outputs

Inputs: training data, domain bounds, groups/time, model family, treatment budget. Outputs: flags with reasons, fitted thresholds/model, treatment policy, and sensitivity report.

## Execution and decision logic

1. Apply domain validity bounds first; quarantine or correct invalid values.
2. Use robust univariate rules (IQR/MAD/quantiles) within meaningful groups when appropriate.
3. Use multivariate detectors only with enough data and validate stability; anomaly does not imply error.
4. Prefer robust scaling or tree models for valid extremes; winsorize/transform only when CV and domain support it.
5. Fit thresholds/detectors on training folds and apply unchanged elsewhere. Never delete rows merely to improve metrics.

## Python example

```python
import numpy as np
import pandas as pd


def mad_flags(series: pd.Series, z_limit: float = 3.5) -> pd.Series:
    """Flag robust univariate extremes while retaining missing values as unflagged."""
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=series.index)
    robust_z = 0.6745 * (values - median).abs() / mad
    return robust_z.gt(z_limit).fillna(False)
```

## Validation checks and best practices

Report flagged counts/rates by split and group, confirm threshold provenance, and test constant/all-null columns. Compare untreated and treated CV results and coefficient stability. Retain flags for audit, avoid test-fitted caps, and monitor changing outlier rates as drift.
