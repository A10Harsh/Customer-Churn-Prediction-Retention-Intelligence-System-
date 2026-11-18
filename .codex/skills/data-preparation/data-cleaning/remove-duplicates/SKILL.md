---
name: remove-duplicates
description: Detect exact, key-based, temporal, and conflicting duplicates and resolve them with auditable entity-aware rules.
---

# Remove Duplicates

## Purpose and use cases

Prevent repeated records or entities from biasing training and crossing splits. Use before splitting for identity analysis, then enforce split isolation after splitting.

## Inputs and outputs

Inputs: data, primary/entity keys, event time, source priority, target contract. Outputs: duplicate groups, conflict report, deterministic resolution policy, retained frame, and lineage mapping.

## Execution and decision logic

1. Count exact row duplicates separately from duplicate keys.
2. For duplicate keys, compare non-key fields and target; classify harmless repeats versus conflicts.
3. Resolve only with a domain rule: latest valid event, trusted source priority, or aggregation matching the prediction unit.
4. Quarantine unresolved target/key conflicts; never keep an arbitrary first row.
5. Recheck that no entity/group appears across train, validation, and test when isolation is required.

## Python example

```python
import pandas as pd


def keep_latest(
    frame: pd.DataFrame, *, keys: list[str], timestamp: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve duplicate keys using the latest timestamp deterministically."""
    ordered = frame.sort_values([*keys, timestamp], kind="stable")
    duplicate_mask = ordered.duplicated(keys, keep=False)
    conflicts = ordered.loc[duplicate_mask].copy()
    retained = ordered.drop_duplicates(keys, keep="last")
    return retained, conflicts
```

## Validation checks and best practices

Assert expected key uniqueness, retained-plus-removed reconciliation, stable results under input reordering, and explicit handling of tied timestamps. Preserve source row IDs and resolution reasons. Fit no rule from test outcomes and avoid fuzzy entity matching without reviewed thresholds.
