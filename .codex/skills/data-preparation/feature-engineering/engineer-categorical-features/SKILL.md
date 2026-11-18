---
name: engineer-categorical-features
description: Build reviewed category groupings, rare-level policies, cross-features, and presence indicators before encoding.
---

# Engineer Categorical Features

## Purpose and use cases

Convert noisy or overly granular categories into stable semantic groupings. Use before encoding when domain taxonomies, rare levels, or meaningful category interactions exist.

## Inputs and outputs

Inputs: training categories, taxonomy/reference mappings, minimum support, prediction context. Outputs: deterministic category transformer, mappings, unknown/rare policy, feature lineage, and coverage report.

## Execution and decision logic

1. Normalize representations before grouping; keep missing distinct from unknown.
2. Prefer external/domain taxonomies over target-driven grouping.
3. Learn rare-level sets on training folds only; map unseen levels to `__UNKNOWN__`, not the rare bucket unless policy says so.
4. Create category crosses only for plausible interactions with adequate support.
5. Avoid target-conditioned groupings outside a properly cross-fitted encoder.

## Python example

```python
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group categories below a training-frequency threshold."""

    def __init__(self, min_frequency: float = 0.01) -> None:
        self.min_frequency = min_frequency

    def fit(self, X: pd.DataFrame, y: object = None) -> "RareCategoryGrouper":
        self.keep_ = {
            c: set(
                X[c]
                .value_counts(normalize=True)
                .loc[lambda s: s >= self.min_frequency]
                .index
            )
            for c in X
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(
            **{c: X[c].where(X[c].isin(self.keep_[c]), "__RARE__") for c in X}
        )
```

## Validation checks and best practices

Check mapping coverage, output cardinality, minimum group support, and deterministic unseen behavior. Persist mappings, test null/unseen/rare cases, monitor category churn, and validate category crosses by CV rather than target prevalence inspection on held-out data.
