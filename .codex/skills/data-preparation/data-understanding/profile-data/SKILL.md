---
name: profile-data
description: Produce a bounded, privacy-aware structural and descriptive dataset profile that drives later cleaning and feature decisions.
---

# Profile Data

## Purpose and use cases

Summarize shape, cardinality, missingness, numeric ranges, categorical frequencies, string lengths, and temporal coverage. Use after ingestion and separately for each training split or production batch.

## Inputs and outputs

Inputs: DataFrame, semantic roles, optional target/group/time columns, sampling budget. Outputs: machine-readable profile, concise human report, and prioritized anomaly candidates; never a transformed dataset.

## Execution

1. Record shape, memory, duplicate columns/rows, and physical dtypes.
2. Compute null and unique counts for all columns.
3. Profile numeric quantiles and finite values; categorical top levels and rare mass; datetime min/max/gaps; text length only unless authorized.
4. Segment missingness and target prevalence by time/group when relevant.
5. Rank issues by severity, affected rows, and downstream risk.

## Decision logic

Use exact summaries when affordable; otherwise use a seeded representative sample and label estimates. Suppress raw values for identifiers/PII. Treat high cardinality according to semantic role, not as an automatic defect. Profile train data for modeling decisions and other splits only for consistency checks.

## Python example

```python
import numpy as np
import pandas as pd


def profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Return column-level profile metrics without exposing row values."""
    rows = len(frame)
    return pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing_count": frame.isna().sum(),
            "missing_rate": frame.isna().mean(),
            "unique_count": frame.nunique(dropna=True),
            "infinite_count": [
                int(np.isinf(frame[c]).sum())
                if pd.api.types.is_numeric_dtype(frame[c])
                else 0
                for c in frame
            ],
            "row_count": rows,
        }
    )
```

## Validation checks and best practices

Reconcile profile counts with source counts; assert rates are in `[0, 1]`; identify all-null, constant, non-finite, and near-unique columns. Persist profile code/config and data version, cap category output, distinguish observed facts from recommendations, and compare profiles only when definitions and windows match.
