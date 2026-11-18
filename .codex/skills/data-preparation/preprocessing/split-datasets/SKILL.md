---
name: split-datasets
description: Create reproducible train, validation, and test partitions that respect time, entities, groups, target prevalence, and final-test isolation.
---

# Split Datasets

## Purpose and use cases

Establish unbiased evaluation partitions before fitting any learned data transform. Use immediately after target/key validation.

## Inputs and outputs

Inputs: features/target, entity/group key, timestamps, split proportions, seed, prediction contract. Outputs: immutable split assignments or indices, split datasets, method/config, and isolation/balance report.

## Execution and decision logic

1. Split by time for future-facing prediction; use groups when repeated entities could cross partitions; otherwise use seeded random splitting.
2. Stratify classification targets when compatible with group/time constraints and minority counts allow it.
3. Create train/test first, then train/validation, preserving the final test untouched.
4. Persist assignments by stable row/entity ID so reruns do not reshuffle.
5. Fit every imputer, encoder, sampler, scaler, selector, and reducer after splitting, on training only.

## Python example

```python
from typing import Any

from sklearn.model_selection import train_test_split


def three_way_split(X: Any, y: Any, *, seed: int = 42) -> tuple[Any, ...]:
    """Create 70/15/15 stratified splits for iid classification data."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
```

## Validation checks and best practices

Assert disjoint stable row IDs and groups, complete row reconciliation, adequate class counts, and temporal ordering/gaps. Compare only basic split distributions, not test-driven preprocessing choices. Store seed, method, hashes, and cutoff; never tune on the test set.
