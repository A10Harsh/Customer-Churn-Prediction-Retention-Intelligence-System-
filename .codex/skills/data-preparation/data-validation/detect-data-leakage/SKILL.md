---
name: detect-data-leakage
description: Audit target, temporal, entity, preprocessing, split, and proxy leakage before training and after feature generation.
---

# Detect Data Leakage

## Purpose and use cases

Prevent invalid offline performance caused by information unavailable at prediction time or shared across evaluation boundaries.

## Inputs and outputs

Inputs: prediction contract, raw/engineered schema, feature lineage, split assignments, pipeline, timestamps/groups. Outputs: leakage findings with evidence/severity, blocked features/steps, and signed-off audit checklist.

## Execution and decision logic

1. Target leakage: find target copies, label-construction inputs, post-outcome statuses, and aggregates containing the outcome.
2. Temporal leakage: verify each source event and aggregation end precede the row's prediction cutoff.
3. Entity leakage: check duplicate entities/households/devices across partitions and unsafe aggregates over future/all data.
4. Preprocessing leakage: ensure split occurs before fitting and every learned step is inside pipeline/CV; samplers touch training folds only.
5. Proxy leakage: investigate implausibly strong single features, suspicious names, near-deterministic associations, and source process artifacts. Block on unresolved high-severity findings.

## Python example

```python
import pandas as pd


def split_overlap(assignments: pd.DataFrame, *, key: str, split: str) -> pd.DataFrame:
    """Return keys assigned to more than one split."""
    counts = assignments.groupby(key, dropna=False)[split].nunique()
    return counts[counts > 1].rename("split_count").reset_index()


def future_feature_rows(frame: pd.DataFrame, event_time: str, cutoff: str) -> pd.Index:
    """Identify features sourced after their prediction cutoff."""
    return frame.index[frame[event_time] > frame[cutoff]]
```

## Validation checks and best practices

Require zero target/ID features, zero forbidden group overlap, zero post-cutoff feature events, and train-only fitted transformer provenance. Test pipeline calls and aggregation windows. Re-run after schema/feature changes and prefer removing uncertain fields over rationalizing suspicious performance.
