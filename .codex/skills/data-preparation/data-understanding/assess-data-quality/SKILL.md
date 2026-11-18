---
name: assess-data-quality
description: Measure completeness, validity, uniqueness, consistency, timeliness, and relational integrity against explicit ML data contracts.
---

# Assess Data Quality

## Purpose and use cases

Create an evidence-based quality gate after ingestion and before model preparation, and reuse it for incoming batches.

## Inputs and outputs

Inputs: dataset, schema, business constraints, keys/relationships, observation window, severity thresholds. Outputs: metric table with numerator/denominator, failed-row references, severity, owner/action, and pass/warn/fail status.

## Execution

1. Translate contracts into column, row, cross-field, temporal, and relational checks.
2. Compute failures and rates without changing raw values.
3. Separate source defects, expected missingness, and sampling artifacts.
4. Prioritize by leakage risk, target/key impact, prevalence, and repairability.
5. Gate downstream work; quarantine correctable records and escalate contract failures.

## Decision logic

Zero tolerance is appropriate for duplicate primary keys, invalid target labels, future-dated observation features, and broken split identity. Configurable tolerances may fit optional fields or known source latency. Never hide poor quality by imputing before measurement.

## Python example

```python
from collections.abc import Callable
import pandas as pd


def quality_check(
    frame: pd.DataFrame, rules: dict[str, Callable[[pd.DataFrame], pd.Series]]
) -> pd.DataFrame:
    """Evaluate rules that return True for valid rows."""
    records = []
    for name, rule in rules.items():
        valid = rule(frame).fillna(False)
        failures = int((~valid).sum())
        records.append(
            {"rule": name, "failed": failures, "rate": failures / max(len(frame), 1)}
        )
    return pd.DataFrame(records)
```

## Validation checks and best practices

Ensure each metric has a denominator and reproducible query, failed-row references are stable, and total rows reconcile. Test boundary cases and null semantics. Version rules and thresholds, retain issue history, avoid aggregate-only reporting, and rerun after every source/schema change.
