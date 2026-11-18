---
name: stratified-sampling
description: Preserve target or joint subgroup proportions in splits or bounded analysis samples while respecting minimum counts and group constraints.
---

# Stratified Sampling

## Purpose and use cases

Create representative classification partitions or analysis samples when random sampling could omit rare classes/cohorts.

## Inputs and outputs

Inputs: frame, one or more strata, sample size/fractions, group/time constraints, minimum cell size, seed. Outputs: sampled indices, inclusion probabilities/weights when needed, and balance report.

## Execution and decision logic

1. Choose strata tied to evaluation needs, usually target and optionally a small key subgroup.
2. Collapse or explicitly handle cells too small for requested partitions; never silently duplicate rows.
3. If entities repeat, stratify at entity/group level or use StratifiedGroupKFold.
4. Preserve time order when temporal validity dominates prevalence matching.
5. Store sample indices and seed; use sampling weights for population estimates after disproportionate sampling.

## Python example

```python
import pandas as pd


def stratified_sample(
    frame: pd.DataFrame, by: list[str], fraction: float, seed: int = 42
) -> pd.DataFrame:
    """Sample the same fraction within each observed stratum."""
    return (
        frame.groupby(by, group_keys=False, dropna=False)
        .sample(frac=fraction, random_state=seed)
        .sort_index()
    )
```

## Validation checks and best practices

Compare source/sample counts and proportions with tolerances, assert unique sampled row IDs, and report empty/undersized cells. Avoid excessive joint strata, retain inclusion probabilities, and never use oversampling as a substitute for evaluation-set representativeness.
