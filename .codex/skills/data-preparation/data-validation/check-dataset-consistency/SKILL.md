---
name: check-dataset-consistency
description: Verify schema, row identity, label, units, category, and preprocessing consistency across sources, splits, reruns, and serving batches.
---

# Check Dataset Consistency

## Purpose and use cases

Catch train/validation/test or train/serve mismatches that are not necessarily statistical drift.

## Inputs and outputs

Inputs: two or more datasets/manifests, contracts, split assignments, keys, pipeline metadata. Outputs: structured comparison, overlap/mismatch tables, severity status, and remediation recommendations.

## Execution and decision logic

1. Compare source/version manifests, row/column counts, names, semantic dtypes, units, null tokens, target coding, and timezone.
2. Assert split row/group identities are disjoint and union to the eligible population.
3. Compare category vocabularies and ranges to contract, distinguishing expected unseen values from schema breaks.
4. Check preprocessing artifact schema/version and output feature order across batch and single-row execution.
5. Treat schema/unit/label mismatches as consistency failures; route distribution changes to drift analysis.

## Python example

```python
import pandas as pd


def compare_columns(
    reference: pd.DataFrame, current: pd.DataFrame
) -> dict[str, object]:
    """Compare column sets and pandas dtypes."""
    ref, cur = set(reference), set(current)
    shared = ref & cur
    return {
        "missing": sorted(ref - cur),
        "unexpected": sorted(cur - ref),
        "dtype_mismatches": sorted(
            c for c in shared if str(reference[c].dtype) != str(current[c].dtype)
        ),
    }
```

## Validation checks and best practices

Require complete row reconciliation and zero forbidden overlap, and test schema comparisons in both directions. Compare semantic types rather than trusting storage dtypes alone. Persist manifests/hashes, make allowed differences explicit, and never concatenate splits to hide incompatibilities.
