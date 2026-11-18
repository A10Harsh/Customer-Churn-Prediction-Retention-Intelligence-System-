---
name: ingest-data
description: Load tabular files, database extracts, or in-memory frames with explicit parsing, provenance, and non-destructive checks before ML preparation.
---

# Ingest Data

## Purpose and use cases

Create a reproducible, immutable landing representation when receiving CSV, Parquet, JSONL, SQL results, or a DataFrame. Use before profiling or whenever parsing, row identity, encoding, or source-version uncertainty exists.

## Inputs and outputs

Inputs: source URI/path or frame, format, optional schema hints, primary/entity keys, snapshot time, and configuration. Outputs: loaded frame, source manifest (URI, size, checksum/version, extraction time, parser options), and rejected-row report. Never log credentials or raw PII.

## Execution

1. Confirm source authorization and format; preserve the raw source.
2. Read a bounded sample to detect delimiter, encoding, headers, null tokens, and malformed rows. Do not infer dates or identifiers aggressively.
3. Load with explicit options; retain stable source row IDs for traceability.
4. Compare observed row/column counts, keys, and checksum/version with expectations.
5. Write only derived outputs to configured interim storage and record provenance.

## Decision logic

- Prefer Parquet with declared types for large repeatable extracts; chunk CSV/JSONL if memory estimates are unsafe.
- Read identifiers as strings when leading zeros or non-arithmetic semantics are possible.
- Parse timestamps only with an explicit timezone/format; quarantine ambiguous values.
- For SQL, use a read-only, bounded query and stable snapshot. Never mutate `raw` tables.

## Python example

```python
from pathlib import Path
import hashlib
import pandas as pd


def load_csv(
    path: Path, *, id_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a CSV deterministically and return its provenance manifest."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(
        path,
        dtype={column: "string" for column in id_columns},
        na_values=["", "NA", "N/A", "null"],
        keep_default_na=True,
    )
    frame.insert(0, "_source_row", range(1, len(frame) + 1))
    return frame, {
        "path": str(path),
        "sha256": digest,
        "rows": len(frame),
        "columns": frame.columns.tolist(),
    }
```

## Validation checks and best practices

Assert non-empty data, unique column names, stable row counts, expected keys, and no silent skipped lines. Test parsing with quoted delimiters, leading-zero IDs, null tokens, and invalid dates. Keep parser configuration in YAML, checksum inputs, avoid full-data logging, and fail with actionable rejected-row counts.
