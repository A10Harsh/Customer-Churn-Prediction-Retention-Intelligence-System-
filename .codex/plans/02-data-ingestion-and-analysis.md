# Implementation Plan: Data Ingestion and Analysis

The implementation should proceed through nine gated work packages. Each
package must pass its tests before the next begins; no model, feature pipeline,
notebook, Prefect flow, or external deployment work belongs in this phase.

This plan is governed by `AGENTS.md`, `.codex/spec.md`, and
`.codex/spec/02-data-ingestion-and-analysis.md`.

## Required specification alignments

Two files are required by `AGENTS.md` but omitted from the revised spec's file
list:

- `configs/phase2.yaml`: required because paths, split ratios, seed, and
  thresholds cannot be hardcoded.
- `database/schema.sql` and `database/staging.sql`: required because the base
  schemas must be defined in SQL.

`PyYAML` should also be declared directly because production code will read
`configs/phase2.yaml`; it should not rely on DVC's transitive installation.

The existing `.gitignore` and specification changes are user-owned and must be
preserved.

## Work packages

| Package | Main result | Gate before continuing |
| --- | --- | --- |
| 1. Preflight | Verified source and clean baseline | No destination collision; baseline checks recorded |
| 2. Configuration | Dependencies and non-secret contracts | Config validation tests pass |
| 3. DVC setup | Immutable canonical raw dataset | Raw checksum matches; CSV remains outside Git |
| 4. PostgreSQL | Healthy service and four schemas | Non-destructive bootstrap succeeds twice |
| 5. Ingestion | Transactional raw and staging load | Counts match; rerun is a no-op |
| 6. Validation | Strict Pandera contract | Real data passes; bad fixtures fail |
| 7. Splitting | Fixed 70/15/15 partitions | Deterministic, disjoint, exhaustive |
| 8. Analysis | Training-only aggregate report | Output is deterministic and privacy-safe |
| 9. Lineage and verification | Reproducible DVC pipeline | All lint, tests, integration, and DVC checks pass |

## 1. Preflight and safety baseline

Before editing:

1. Inspect `git status --short` and relevant diffs.
2. Preserve the current `.gitignore` and specification changes.
3. Recalculate, rather than copy from the spec:

   - source SHA-256;
   - row and column counts;
   - source header;
   - target counts;
   - blank `TotalCharges` count;
   - duplicate and blank identifier counts.

4. Check whether `data/raw/Telco-Customer-Churn.csv` exists:

   - If absent, it may be copied later.
   - If present with identical bytes, reuse it.
   - If present with different bytes, stop without overwriting it.

5. Run and record the baseline:

```sh
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
```

Do not read `.env`, print records, display customer IDs, or alter the source
file.

## 2. Dependencies, environment, and configuration

### Modify `pyproject.toml`

Add bounded Python 3.11-compatible runtime dependencies:

- `pandas`
- `pandera[pandas]`
- `sqlalchemy`
- `psycopg[binary]`
- `scikit-learn`
- `scipy`
- `dvc`
- `pyyaml`

Add the `integration` pytest marker. Configure normal `pytest` runs to exclude
integration tests, while `pytest -m integration` explicitly selects them.

### Modify `.env.example`

Use names only:

```text
DB_HOST=
DB_PORT=
DB_NAME=
DB_USER=
DB_PASSWORD=
MLFLOW_TRACKING_URI=
ANTHROPIC_API_KEY=
```

Remove the generic `LLM_API_KEY`. Production code must read these values from
the environment and never log them.

### Create `configs/phase2.yaml`

Include:

- canonical source path;
- manifest and report paths;
- exact expected source columns;
- database schema/table names;
- split ratios `0.70/0.15/0.15`;
- seed `42`;
- sparse-category threshold `30`;
- correlation flag threshold `0.8`;
- Cramer's V threshold `0.5`;
- deterministic JSON and CSV settings.

Add configuration validation for:

- ratios summing to one;
- nonnegative thresholds;
- expected column uniqueness;
- safe repository-relative paths;
- required database variable names.

## 3. DVC initialization and raw placement

1. Run `dvc init` locally without a remote.
2. Copy, rather than move, the current CSV to `data/raw/`.
3. Recalculate and compare both files' SHA-256 values.
4. Run:

```sh
.venv/bin/dvc add data/raw/Telco-Customer-Churn.csv
```

5. Verify:

   - the raw CSV is ignored by Git;
   - its `.dvc` metadata is visible to Git;
   - the original file still exists;
   - no remote appears in `.dvc/config`;
   - no customer-bearing CSV appears in `git status`.

No Git commit should be created unless separately requested.

## 4. PostgreSQL and SQL contracts

### Create `docker-compose.yml`

Define one pinned PostgreSQL service with:

- an exact image version rather than a floating `latest` tag;
- `${DB_*:?}` environment-variable requirements;
- host binding to `127.0.0.1`;
- a `pg_isready` health check;
- a persistent named volume;
- no embedded credentials;
- no destructive initialization or teardown command.

### Create `database/schema.sql`

Use only non-destructive statements:

- `CREATE SCHEMA IF NOT EXISTS` for `raw`, `staging`, `features`, and
  `predictions`;
- `raw.ingestion_log`, keyed by `source_version`;
- `raw.telco_customers`, with every source field stored as `TEXT`;
- `staging.telco_customers`, with typed lowercase columns and primary key
  `(source_version, customerid)`.

Raw records must only be inserted for a previously unseen source version. No
raw update or delete path should exist.

### Create `database/staging.sql`

Implement an explicit raw-to-staging insert:

- `Churn`: `Yes -> 1`, `No -> 0`;
- `TotalCharges`: blank or whitespace -> `NULL`, otherwise numeric;
- typed `SeniorCitizen`, `tenure`, and `MonthlyCharges`;
- all source rows retained;
- parameterized `source_version`;
- no interpolation of SQL values;
- no `DROP`, `TRUNCATE`, or `DELETE`.

Run bootstrap twice to prove repeatability.

## 5. Implement `src/churn_phase2.py`

Keep the spec's single-module design, but divide it into small typed functions.

### Configuration and utilities

- `Phase2Config`
- `load_config()`
- `require_database_environment()`
- `sha256_file()`
- `write_json_atomic()`
- `write_csv_atomic()`

### Source handling

- `read_source_csv()`
- `validate_source_header()`
- `validate_ingestion_preconditions()`

Pre-ingestion checks must reject:

- malformed or short rows;
- missing or duplicate IDs;
- unknown target labels;
- invalid required numerics.

A bad categorical value should not be rejected here; it must reach staging and
fail Pandera validation as required by the failure-path test.

### Database operations

- `create_engine_from_environment()`
- `bootstrap_database()`
- `ingest_source()`
- `load_staged_frame()`

The ingestion transaction should:

1. Parse and validate the source before opening the transaction.
2. Insert the `ingestion_log` row with `ON CONFLICT DO NOTHING RETURNING`.
3. If the version already exists, verify stored counts and return without
   mutation.
4. Insert all raw rows.
5. Execute the staging transformation.
6. Verify raw, staged, and source row counts are identical.
7. Roll back the entire transaction on any failure.
8. Commit before atomically writing the manifest.

Logs may contain counts and a short source-version prefix, never records or
identifiers.

### Validation

- `build_staging_schema()`
- `validate_staging()`
- `build_structure_profile()`

Pandera should use `strict=True` and `lazy=True`. Its failure report should
contain only:

- column;
- check name;
- failure count.

Cap the grouped output at 50 entries and exclude failure values, row indexes,
and row contents.

The structure profile can contain shape, dtypes, missing counts, and unique-ID
count. It must not perform full-dataset target analysis.

### Splitting

- `create_partitions()`
- `validate_partitions()`
- `write_partitions()`

Algorithm:

1. Sort staged rows by `customerid`.
2. Split 70% train and 30% temporary using stratification and seed 42.
3. Split the temporary set equally into validation and test using
   stratification and seed 42.
4. Sort each partition before serialization for byte stability.
5. Verify disjoint IDs, complete coverage, expected sizes, and target
   prevalence within integer-rounding tolerance.
6. Write outputs only after all assertions pass.

Do not drop `customerid` from the protected split artifacts; Phase 3 will
exclude it from model features.

### Analysis

- `numeric_analysis()`
- `categorical_analysis()`
- `corrected_cramers_v()`
- `build_analysis_metrics()`
- `render_analysis_report()`

The analysis command must accept only:

- `structure_profile.json`;
- `train.csv`;
- configuration.

It must never read validation or test partitions.

Exclude `customerid` from every analysis table. Use pairwise non-null
observations for numeric correlation; do not impute `TotalCharges`.
Categoricals should report aggregate level counts, churn rates, Cramer's V, and
sparse-level flags.

Proxy flags are review evidence only. They must not automatically remove or
select features.

### CLI

Use `argparse` with:

```text
bootstrap
ingest
validate
split
analyze
all
```

Catch only known phase-specific exceptions at the CLI boundary. Return nonzero
exit codes for validation and contract failures without suppressing the
underlying cause.

## 6. Deterministic artifacts

Use these locations:

```text
data/processed/ingest_manifest.json
data/processed/validation_report.json
data/processed/structure_profile.json
data/interim/train.csv
data/interim/validation.csv
data/interim/test.csv
data/processed/split_manifest.json
data/processed/analysis_metrics.json
docs/data-analysis-report.md
```

Every manifest should include the full source version. Avoid timestamps and
machine-specific paths so reruns remain byte-stable.

Serialization rules should include:

- sorted JSON keys;
- fixed indentation;
- explicit UTF-8;
- normalized newline;
- stable column order;
- stable row ordering;
- finite JSON numbers only;
- documented float rounding.

## 7. DVC lineage

Create four stages in `dvc.yaml`:

1. `ingest`
2. `validate`
3. `split`
4. `analyze`

Each stage should declare:

- `src/churn_phase2.py`;
- relevant SQL files;
- `configs/phase2.yaml`;
- upstream manifests/artifacts;
- required outputs;
- relevant config parameters.

The analyze stage must depend only on `train.csv`, `structure_profile.json`,
code, and configuration, not validation or test data.

Because PostgreSQL is external state, document this limitation clearly:

- stages always select the manifest's exact `source_version`;
- no stage selects "latest" database rows;
- if the PostgreSQL volume is lost, run `dvc repro -f ingest`;
- never remove the volume automatically.

Run `dvc repro` once to generate `dvc.lock`, then a second time to verify there
are no changes.

## 8. Tests

Suggested test layout:

```text
tests/data/test_config.py
tests/data/test_source_contract.py
tests/data/test_staging_schema.py
tests/data/test_split.py
tests/data/test_analysis.py
tests/data/test_reporting_privacy.py
tests/data/test_cli.py
tests/integration/test_postgres_ingestion.py
```

Use synthetic records and fake identifiers in unit tests.

Required test groups:

- source header and row-width checks;
- immutable raw values;
- exact target mapping;
- blank `TotalCharges -> NULL`;
- known-bad Pandera cases;
- deterministic and disjoint splits;
- no learned preprocessing;
- analysis restricted to training data;
- Cramer's V edge cases;
- deterministic JSON, CSV, and Markdown;
- no identifiers or secrets in logs/reports;
- duplicate-ID transactional rollback;
- unknown-target transactional rollback;
- invalid category passes ingestion but blocks splitting at validation;
- repeat ingestion is a no-op;
- four PostgreSQL schemas exist.

## 9. Documentation and final verification

Update `README.md` with:

- prerequisites;
- environment variable names;
- exact bootstrap and stage commands;
- artifact locations;
- DVC lineage;
- rerun semantics;
- stale-database-manifest limitation;
- safe troubleshooting;
- a warning not to run `docker compose down -v`.

Do not add measured values until the commands have actually generated them.

Final execution order:

```sh
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
docker compose config
docker compose up -d postgres
.venv/bin/pytest -m integration
.venv/bin/dvc dag
.venv/bin/dvc repro
.venv/bin/dvc repro
.venv/bin/pre-commit run --all-files
git status --short
```

Completion requires actual command output, not expected results.

## Stop conditions

Stop and request direction if:

- the canonical raw destination contains different bytes;
- an existing PostgreSQL volume contains incompatible project tables;
- schema creation would require destructive SQL;
- an applied migration would need editing;
- dependency installation requires unapproved external access;
- a DVC-tracked artifact would be overwritten or deleted;
- measured source evidence differs materially from the expected contract;
- any validation, leakage, privacy, or transactional test fails.
