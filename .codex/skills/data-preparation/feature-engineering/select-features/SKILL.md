---
name: select-features
description: Remove unsafe or unusable features and perform filter, embedded, or wrapper selection strictly inside training folds.
---

# Select Features

## Purpose and use cases

Reduce leakage, cost, redundancy, instability, or dimensionality while retaining predictive signal and deployability.

## Inputs and outputs

Inputs: training pipeline/data, target, groups/time, feature lineage, estimator, CV design, feature costs. Outputs: selected names/mask, fitted selector in pipeline, exclusion reasons, stability and CV comparison.

## Execution and decision logic

1. Deterministically exclude target, IDs, post-outcome fields, constants, unavailable and prohibited features.
2. Remove exact duplicates and review highly redundant features by availability/cost/stability.
3. Use univariate filters for cheap screening, L1/tree importance for embedded selection, or RFE/sequential methods only when compute and CV design support them.
4. Fit selector within each training fold; never select once on all data before CV.
5. Prefer the smallest stable set within acceptable validated performance and business constraints.

## Python example

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


def embedded_selector(seed: int = 42) -> SelectFromModel:
    """Build an L1 selector to be fitted inside a pipeline/CV fold."""
    model = LogisticRegression(
        penalty="l1", solver="liblinear", class_weight="balanced", random_state=seed
    )
    return SelectFromModel(model, threshold="median")
```

## Validation checks and best practices

Confirm no excluded fields survive, selection is fold-local, and validation/test were transform-only. Report selection frequency across folds and feature-count/performance tradeoff. Preserve lineage and feature ordering; do not interpret impurity importance as causal evidence.
