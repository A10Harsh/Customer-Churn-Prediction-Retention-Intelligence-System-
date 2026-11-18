---
name: correct-invalid-data
description: Repair or quarantine contract-invalid values using deterministic domain rules and complete correction lineage.
---

# Correct Invalid Data

## Purpose and use cases

Handle impossible ranges, broken cross-field relationships, malformed codes, and invalid event order after quality rules identify them.

## Inputs and outputs

Inputs: data, versioned constraints, trusted reference tables, correction rules, severity policy. Outputs: corrected and quarantined datasets, reason codes, before/after values or hashes, and quality rerun.

## Execution and decision logic

1. Re-evaluate each failed rule and confirm the contract is correct.
2. Auto-correct only deterministic, lossless cases backed by trusted evidence (for example, known unit conversion).
3. Set to missing only when the downstream missing policy explicitly handles invalid-as-unknown.
4. Quarantine ambiguous target, key, temporal, or cross-field failures; request source remediation.
5. Re-run the full applicable quality suite and preserve correction lineage. Never edit immutable raw data.

## Python example

```python
from collections.abc import Callable
import pandas as pd


def apply_corrections(
    frame: pd.DataFrame,
    rules: list[
        tuple[
            str,
            Callable[[pd.DataFrame], pd.Series],
            Callable[[pd.DataFrame], pd.DataFrame],
        ]
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply reviewed corrections and return row/reason audit records."""
    result = frame.copy()
    audit: list[dict[str, object]] = []
    for name, invalid, correction in rules:
        mask = invalid(result).fillna(False)
        audit.extend({"index": idx, "rule": name} for idx in result.index[mask])
        result.loc[mask] = correction(result.loc[mask])
    return result, pd.DataFrame(audit)
```

## Validation checks and best practices

Assert corrections are idempotent, constrained to flagged rows/columns, and do not alter target semantics or row identity. Reconcile audit counts, test boundaries, and fail if invalids remain above policy. Keep rules small, typed, reviewed, and configured; never use broad exception handling to conceal failures.
