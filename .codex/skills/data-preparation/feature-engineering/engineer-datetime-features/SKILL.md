---
name: engineer-datetime-features
description: Create calendar, age, tenure, recency, duration, and cyclic features with explicit timezone and prediction-cutoff semantics.
---

# Engineer Datetime Features

## Purpose and use cases

Extract temporal signal while preventing future information. Use for events, subscriptions, cohorts, seasonality, and recency/frequency behavior.

## Inputs and outputs

Inputs: timezone-aware timestamps, prediction cutoff per row, event history, business calendar. Outputs: deterministic temporal transformer, feature lineage, invalid-order report, and model-ready features.

## Execution and decision logic

1. Normalize timestamps to the declared timezone and preserve source timezone metadata.
2. Calculate age/tenure/recency relative to each row's prediction cutoff, never current wall-clock time.
3. Aggregate only events at or before cutoff; use closed-window definitions.
4. Encode periodic fields with sine/cosine when continuity matters; retain calendar categories for trees if useful.
5. Treat future/negative durations as quality failures, not clipped values.

## Python example

```python
import numpy as np
import pandas as pd


def add_time_features(frame: pd.DataFrame, event: str, cutoff: str) -> pd.DataFrame:
    """Create leakage-safe recency and cyclic month features."""
    out = frame.copy()
    out["recency_days"] = (out[cutoff] - out[event]).dt.total_seconds() / 86400
    month = out[event].dt.month
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return out
```

## Validation checks and best practices

Assert timezone consistency, nonnegative valid durations, cutoff filtering, and boundary behavior at midnight/DST/leap dates. Test historical replay, store window definitions in config, and audit source timestamps that occur after prediction time.
