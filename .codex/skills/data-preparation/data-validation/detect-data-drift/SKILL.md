---
name: detect-data-drift
description: Compare reference and current feature distributions with type-aware tests, effect sizes, multiplicity controls, and operational context.
---

# Detect Data Drift

## Purpose and use cases

Detect covariate, missingness, category, and target-prevalence changes between training/reference data and later cohorts or production batches.

## Inputs and outputs

Inputs: reference/current datasets, feature contract, time windows, minimum sample sizes, thresholds, segment definitions. Outputs: per-feature drift metrics and counts, corrected test results, severity, plots, and investigation summary.

## Execution and decision logic

1. Confirm schema and comparable population/window before statistical comparison.
2. Numeric: PSI/Wasserstein/KS plus quantile and missingness changes; categorical: Jensen-Shannon/chi-square plus new-category mass; target drift only after labels mature.
3. Report effect size and practical threshold, not p-value alone; large samples make trivial shifts significant.
4. Correct multiple comparisons and inspect critical segments/time trends.
5. Drift triggers investigation, not automatic retraining; check source/process changes and performance when labels arrive.

## Python example

```python
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def numeric_drift(reference: pd.Series, current: pd.Series) -> dict[str, float]:
    """Return robust numeric drift facts on finite observations."""
    ref = (
        pd.to_numeric(reference, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    cur = (
        pd.to_numeric(current, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return {
        "wasserstein": float(wasserstein_distance(ref, cur)),
        "missing_delta": float(current.isna().mean() - reference.isna().mean()),
        "reference_n": float(len(ref)),
        "current_n": float(len(cur)),
    }
```

## Validation checks and best practices

Require minimum sample counts, compatible units/schema, stable binning derived from reference only, and reproducible windows. Test no-drift and known-shift fixtures. Store thresholds/config, report actual metrics, avoid sensitive subgroup exposure, and separate drift from data-quality failure.
