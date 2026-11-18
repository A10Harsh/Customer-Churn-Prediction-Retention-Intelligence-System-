---
name: normalize-inconsistent-data
description: Canonicalize inconsistent categories, units, strings, booleans, and timestamps using explicit reversible mappings.
---

# Normalize Inconsistent Data

## Purpose and use cases

Resolve representation differences such as whitespace/case, aliases, mixed units, boolean spellings, and timezone formats without erasing semantic distinctions.

## Inputs and outputs

Inputs: data, semantic schema, canonical dictionaries, units/timezone, locale. Outputs: normalized frame, mapping/version, unknown-value report, and row-level change log.

## Execution and decision logic

1. Inventory distinct representations and their counts; protect identifiers and free text.
2. Apply safe lexical normalization (Unicode form, trim) only where semantics permit.
3. Map aliases through reviewed dictionaries and convert units with explicit source-unit evidence.
4. Normalize timestamps to a declared timezone while retaining event semantics.
5. Map unseen/ambiguous values to an explicit unknown state or quarantine; never guess by nearest spelling in production.

## Python example

```python
import pandas as pd


def normalize_categories(
    series: pd.Series, aliases: dict[str, str]
) -> tuple[pd.Series, pd.Series]:
    """Normalize category spelling and report unmapped observed values."""
    keys = series.astype("string").str.normalize("NFKC").str.strip().str.casefold()
    normalized = keys.map(aliases).astype("string")
    unknown = series.notna() & normalized.isna()
    return normalized.fillna("__UNKNOWN__"), unknown
```

## Validation checks and best practices

Check idempotence, row-count preservation, no unintended ID changes, unit round trips, and known mapping coverage. Version dictionaries in configuration, retain original values or source row lineage, test locale/timezone boundaries, and distinguish `unknown` from `missing`.
