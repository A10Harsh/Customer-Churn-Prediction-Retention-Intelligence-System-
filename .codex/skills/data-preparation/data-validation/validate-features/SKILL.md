---
name: validate-features
description: Enforce feature contracts for names, types, ranges, nullability, categories, finiteness, lineage, and prediction-time availability.
---

# Validate Features

## Purpose and use cases

Gate raw and transformed features before fit, batch scoring, and online inference.

## Inputs and outputs

Inputs: DataFrame or model matrix, versioned feature contract, feature names/lineage, mode (train/serve), severity policy. Outputs: structured validation report, failed row/column references, pass/fail result, and optional quarantine set.

## Execution and decision logic

1. Validate required/extra columns, uniqueness, order policy, semantic and physical types.
2. Check nullability, allowed categories, bounds, regex/units, and cross-feature constraints.
3. After preprocessing, check matrix shape, feature-name alignment, finiteness, sparsity/density, constants, and numeric precision.
4. Verify all features have lineage and are available at prediction time.
5. Fail on missing required features, non-finite model inputs, target/ID presence, or contract-breaking types; warn only under explicit tolerance.

## Python example

```python
import numpy as np
import pandas as pd


def validate_numeric_matrix(frame: pd.DataFrame, expected: list[str]) -> list[str]:
    """Return blocking model-matrix validation errors."""
    errors: list[str] = []
    if frame.columns.tolist() != expected:
        errors.append("feature names/order mismatch")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame):
        errors.append("non-numeric feature")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        errors.append("non-finite values")
    return errors
```

## Validation checks and best practices

Test valid, boundary, missing, extra, wrong-type, unseen-category, and non-finite fixtures. Reconcile failures and quarantine counts. Version contracts with artifacts, use readable reason codes, and never silently reorder or coerce incompatible inputs at serving time.
