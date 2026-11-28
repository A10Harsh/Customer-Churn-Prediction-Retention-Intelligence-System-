# Data Ingestion and Validation Pipeline

## Objective

Establish a reproducible and validated data foundation for the Customer Retention Intelligence System.

## Data Source

IBM Telco Customer Churn dataset, consisting of 21 columns and approximately 7,043 customer records in the standard dataset version.

## Pipeline Stages

### 1. Ingestion

- Read the raw CSV dataset.
- Normalize leading and trailing whitespace.
- Convert blank strings into explicit missing values.
- Preserve the original raw dataset.
- Store ingested data in Parquet format.

### 2. Validation

- Validate required columns and data types.
- Enforce allowed categorical values.
- Validate numerical ranges and finite values.
- Detect duplicate customer identifiers.
- Validate phone and internet service consistency.
- Reject unexpected schema violations.

## Missing Value Policy

Missing TotalCharges values are replaced with zero only when customer tenure is zero and the corresponding correction policy is enabled.

This is a documented assumption for the historical IBM Telco dataset. Other missing values are rejected rather than silently imputed.

## Data Quality Reporting

The pipeline generates a JSON report containing:

- Dataset dimensions.
- Missing-value counts.
- Duplicate counts.
- Data types.
- Applied corrections.
- Validation errors.
- Final validation status.

## Reproducibility

DVC manages dataset versions, stage dependencies, configuration parameters and generated artifacts.

Inspect the pipeline using:

```bash
dvc status
dvc repro --dry
.venv/bin/python scripts/check_deps_consistency.py
```

Phase 9B replaced the global environment lock dependency in `dvc.yaml` with generated manifests in `dvc/deps-*.txt`. The complete resolved environment remains in `requirements-lock.txt`; regenerate the stage manifests after a relevant dependency change with `.venv/bin/python scripts/check_deps_consistency.py --write`. The checker includes active transitive requirements and runtime metadata, then fails when a manifest is stale.

Historical `dvc.lock` still records the former global lock dependency. All seven stages therefore show changed dependencies, and the dry run includes protected final training and held-out evaluation. Do not run unrestricted reproduction while the frozen provenance migration is unresolved. The historical hashes and exact blocker are documented in [Phase 9B DVC audit](genai_nba.md).

## Data Leakage Prevention

This phase performs deterministic data cleaning only. Statistical imputation, feature selection, scaling, encoding and resampling will be performed within the appropriate training pipelines in later phases.

## Testing

The pipeline is validated using Pytest unit and integration tests covering valid inputs, malformed records, duplicate identifiers, invalid categories, business rule violations and failed pipeline execution.
