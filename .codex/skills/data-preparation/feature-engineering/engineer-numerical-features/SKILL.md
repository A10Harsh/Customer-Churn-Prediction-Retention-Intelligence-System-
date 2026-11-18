---
name: engineer-numerical-features
description: Create domain-grounded numeric ratios, differences, aggregates, bins, and interactions with safe arithmetic and prediction-time constraints.
---

# Engineer Numerical Features

## Purpose and use cases

Express stable domain relationships that raw magnitudes miss. Use for rates, recency, utilization, trends, and interactions supported by the prediction contract.

## Inputs and outputs

Inputs: training data, numeric schema, domain formulas, observation cutoff, grouping keys. Outputs: deterministic transformer, feature definitions/lineage, generated matrix, and validation report.

## Execution and decision logic

1. State the hypothesis and verify every source value is available at prediction time.
2. Prefer interpretable ratios/differences/aggregates; define units and denominator policy.
3. Use bins for known nonlinear thresholds or interpretability, with fixed domain edges or train-fitted edges.
4. Add interactions selectively; regularize or validate to avoid feature explosion.
5. Implement identical train/serve logic and fit any learned boundaries on training folds only.

## Python example

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Create safe ratio features from configured numerator/denominator pairs."""

    def __init__(self, pairs: dict[str, tuple[str, str]]) -> None:
        self.pairs = pairs

    def fit(self, X: pd.DataFrame, y: object = None) -> "RatioFeatures":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for name, (num, den) in self.pairs.items():
            out[name] = out[num].div(out[den].replace(0, np.nan))
        return out
```

## Validation checks and best practices

Assert finite outputs or explicit missingness, valid units, stable names, and no source mutation. Test zero/negative denominators and cutoff boundaries. Keep feature lineage, avoid arbitrary polynomial expansion, and validate incremental value with leakage-safe CV.
