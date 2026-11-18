---
name: detect-schema
description: Infer and review an explicit dataset contract containing column roles, constraints, keys, and prediction-time availability.
---

# Detect Schema

## Purpose and use cases

Turn observed columns into a reviewed contract for validation and train/serve parity. Use when onboarding data, changing sources, or discovering undocumented columns.

## Inputs and outputs

Inputs: representative frame, target and prediction contract, domain hints, known keys. Outputs: versioned schema proposal with physical dtype, nullable flag, semantic role, allowed/range constraints, key status, and availability time; plus uncertainty notes.

## Execution

1. Inventory names, dtypes, missingness, cardinality, and examples redacted as needed.
2. Classify target, identifier, group, timestamp, numeric, categorical, boolean, free text, and ignored columns.
3. Infer conservative constraints; do not turn sample minima/maxima into hard domain bounds.
4. Mark each feature's source event time and whether it is available at prediction time.
5. Require human/domain review for ambiguous roles and persist the approved contract.

## Decision logic

Names are hints, not proof. Near-unique integers may be IDs; low-cardinality integers may be categories; string numerics need parse-rate and leading-zero checks. Reject target/proxy/post-outcome fields from features even if predictive.

## Python example

```python
import pandas as pd


def propose_roles(frame: pd.DataFrame) -> dict[str, str]:
    """Propose semantic roles for review; never auto-approve them."""
    roles: dict[str, str] = {}
    for name in frame:
        series = frame[name]
        ratio = series.nunique(dropna=True) / max(series.notna().sum(), 1)
        if name.lower().endswith(("_id", "id")) and ratio > 0.95:
            roles[name] = "identifier"
        elif pd.api.types.is_datetime64_any_dtype(series):
            roles[name] = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            roles[name] = "numeric"
        else:
            roles[name] = (
                "categorical" if series.nunique() <= 100 else "text_or_identifier"
            )
    return roles
```

## Validation checks and best practices

Require unique names, declared target/key/time roles, compatible nullability, and an explicit exclusion list. Test schema validation on valid and intentionally invalid fixtures. Version contracts, allow additive changes only by policy, record uncertainty, and never silently coerce a breaking change.
