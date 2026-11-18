---
name: target-analysis
description: Audit target definition, labels, prevalence, timing, censoring, and subgroup stability before supervised model preparation.
---

# Target Analysis

## Purpose and use cases

Establish whether the supervised outcome is valid and learnable. Use before target-aware EDA, splitting, imbalance treatment, or modeling.

## Inputs and outputs

Inputs: target column, positive-class definition, entity and event times, prediction/observation/outcome windows, groups. Outputs: target audit, invalid/unknown label report, prevalence with counts, temporal/subgroup summaries, and leakage/censoring risks.

## Execution and decision logic

1. Verify label semantics, allowed values, unit of analysis, and label construction code.
2. Check missing/unknown labels, duplicates, impossible event order, delayed labels, and right censoring.
3. Report class counts/rates overall and by time/group with uncertainty for small cells.
4. Confirm target and target-derived fields are excluded from features.
5. If prevalence changes materially, investigate cohort/source effects before resampling. Do not repair ambiguous labels by guessing.

## Python example

```python
import pandas as pd


def binary_target_audit(
    target: pd.Series, allowed: set[object] = {0, 1}
) -> dict[str, object]:
    """Audit a binary target without fabricating unknown labels."""
    observed = set(target.dropna().unique())
    invalid = observed - allowed
    counts = target.value_counts(dropna=False).to_dict()
    return {
        "counts": counts,
        "invalid_values": sorted(map(str, invalid)),
        "missing": int(target.isna().sum()),
        "positive_rate": float((target == 1).mean()),
    }
```

## Validation checks and best practices

Require zero invalid labels, documented unknown/censored handling, and event-time ordering. Recompute labels from source logic on a sample. Persist label version and cutoff; never use test labels to tune preprocessing, class thresholds, or feature selection.
