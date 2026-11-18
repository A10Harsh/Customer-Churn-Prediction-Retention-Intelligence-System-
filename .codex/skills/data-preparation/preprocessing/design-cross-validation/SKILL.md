---
name: design-cross-validation
description: Select reproducible iid, stratified, grouped, repeated, or time-series cross-validation matching the deployment data-generating process.
---

# Design Cross-Validation

## Purpose and use cases

Estimate generalization and tune pipelines without violating entity or temporal independence.

## Inputs and outputs

Inputs: training data/target, groups/time, deployment scenario, class counts, compute budget, seed. Outputs: splitter object or explicit folds, rationale, fold diagnostics, and scoring plan.

## Execution and decision logic

- IID classification: StratifiedKFold; IID regression: KFold.
- Repeated entities/sites: GroupKFold or StratifiedGroupKFold.
- Forecast/future cohorts: TimeSeriesSplit or custom expanding/rolling splits with embargo/gap.
- Very small data: repeated CV or bootstrap uncertainty, while retaining final test isolation.
- Put the complete learned pipeline inside CV. Choose folds so every validation fold has enough positive/negative cases for planned metrics.

## Python example

```python
from typing import Any

from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


def choose_cv(kind: str, folds: int = 5, seed: int = 42) -> Any:
    """Return a splitter aligned with the declared data structure."""
    if kind == "time":
        return TimeSeriesSplit(n_splits=folds, gap=1)
    if kind == "group":
        return GroupKFold(n_splits=folds)
    if kind == "stratified_group":
        return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
```

## Validation checks and best practices

Assert train/validation disjointness by row and group, correct temporal order, complete fold coverage, and viable target counts. Persist fold assignments and seed. Report fold-level and aggregate metrics; do not use test data for CV or threshold selection.
