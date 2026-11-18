---
name: infer-data-types
description: Infer semantic types from physical values using parse evidence and domain context while protecting identifiers and ambiguous fields.
---

# Infer Data Types

## Purpose and use cases

Resolve object/string or overloaded numeric columns before schema approval. Use for CSV imports, mixed values, dates, booleans, currency, percentages, and encoded categories.

## Inputs and outputs

Inputs: DataFrame, locale/timezone, null tokens, domain hints, confidence thresholds. Outputs: per-column proposed type, parse success rate, ambiguity flags, failed examples or row IDs, and explicit conversion plan.

## Execution

1. Normalize only recognized null tokens in a copy.
2. Test candidate parsers without mutating data: boolean, integer/float, datetime, categorical, text.
3. Check leading zeros, units, locale separators, range, cardinality, and mixed timezone behavior.
4. Select a type only above configured confidence and semantic review; otherwise retain string and flag.
5. Convert with failure capture rather than silent `errors="coerce"`.

## Decision logic

Protect identifiers from numeric conversion. Prefer nullable pandas dtypes. Treat 0/1 as boolean only with domain support. Do not parse integers as epoch timestamps without explicit units. Category dtype is useful for stable finite vocabularies, not arbitrary high-cardinality text.

## Python example

```python
import pandas as pd


def numeric_parse_report(series: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """Parse numeric strings and return values, failure mask, and success rate."""
    source = series.astype("string").str.strip()
    parsed = pd.to_numeric(source, errors="coerce")
    attempted = source.notna()
    failed = attempted & parsed.isna()
    rate = float((attempted & parsed.notna()).sum() / max(attempted.sum(), 1))
    return parsed, failed, rate
```

## Validation checks and best practices

Compare non-null counts before/after conversion, assert failed parses are reported, verify round trips for identifiers and exact integers, and check timezone awareness. Store locale, format, units, and confidence thresholds in config; sample failures safely; require review for ambiguous high-impact columns.
