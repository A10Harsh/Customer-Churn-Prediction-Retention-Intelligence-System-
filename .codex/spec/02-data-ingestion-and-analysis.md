# Spec: Data Ingestion and Analysis (Phase 2)

## Metadata

- Step: 02
- Status: Proposed (revised)
- Branch: Not requested
- Governing requirements: `AGENTS.md`, `.codex/spec.md`
- Roadmap phase: Phase 2, Database and ingestion, extended by the requested data-analysis deliverables

## Objective

Ingest the IBM Telco Customer Churn source without changing it, materialize a typed staging dataset in PostgreSQL, validate it with Pandera, create one reproducible split, and generate aggregate data-quality and training-only exploratory artifacts.

This phase builds the data foundation for feature engineering and modeling. It must not train a model, fit learned preprocessing, use validation or test features for analytical decisions, or expose customer identifiers or raw rows in reports or logs.

Deliverables:

- a DVC-tracked raw dataset at `data/raw/`;
- repeatable PostgreSQL bootstrap, raw ingestion, and staging transformation;
- Pandera validation of the staged dataset;
- fixed stratified train/validation/test artifacts (seed 42);
- deterministic aggregate profiling and training-only exploratory analysis;
- tests, DVC lineage, and a measured analysis report.

## Entry gate

### Evidence reported by the spec author (re-verify before implementing)

- Phase 1 scaffold is on `main` at commit `ccd9dd1` with a clean working tree.
- Python 3.11, DVC 3.x, Docker, and Docker Compose are available locally.
- The source CSV exists at `data/Telco-Customer-Churn.csv` and contains `customerID`, `Churn`, and `TotalCharges`.

### Source observations (current local file only)

- 7,043 data rows and 21 columns.
- Target counts: 5,174 `No` and 1,869 `Yes`.
- 11 blank `TotalCharges` values and no other blank cells.
- No duplicate or blank customer IDs.

These numbers are context only. Implementation must re-measure them from the DVC-tracked file, and no generated report may copy them from this document.

### Missing prerequisites delivered by this step

- No `data/raw/` copy, DVC metadata, `dvc.yaml`, or `dvc.lock`.
- No `docker-compose.yml`, schema SQL, ingestion code, Pandera contract, or data tests.
- `.env.example` lacks database variable names and uses `LLM_API_KEY` instead of `ANTHROPIC_API_KEY`.

### Decision

Ready. Stop rather than overwrite data if `data/raw/Telco-Customer-Churn.csv` already exists with different bytes.

## Scope

### In scope

- Initialize local DVC without a remote.
- Copy the source byte-for-byte to the canonical raw path and track it with DVC metadata only. Keep the original file unless the user separately approves removal.
- Add a pinned PostgreSQL service with a health check and persistent named volume.
- Create the `raw`, `staging`, `features`, and `predictions` schemas and the raw/staging tables with non-destructive, repeatable SQL.
- Load source values unchanged into `raw`, and make reruns idempotent.
- Build `staging` with `Churn` mapped to integer 1/0 and blank `TotalCharges` stored as NULL, without dropping any row.
- Validate the staged data with Pandera before any downstream artifact is produced.
- Create the single stratified 70/15/15 split with seed 42.
- Produce full-dataset structural evidence (target not used) and training-only EDA.
- Produce a machine-readable metrics file and a human-readable report, both aggregate-only.
- Add unit and PostgreSQL integration tests.

### Out of scope

- Feature engineering, learned encoding, imputation, scaling, or feature selection (Phase 3).
- Model training, cross-validation, tuning, calibration, thresholds, test evaluation, MLflow, or registration (Phase 4).
- SHAP, GenAI, APIs, Prefect, monitoring, dashboards, deployment, or cloud resources.
- Any feature or modeling decision drawn from validation or test features.
- Editing or deleting the supplied raw file, committing customer data to Git, or printing identifiers.
- Destructive SQL, DVC remotes or pushes, paid calls, or deployments.
- Notebooks.

## Data and artifact contracts

### Raw input

- Path: `data/raw/Telco-Customer-Churn.csv`, versioned by DVC metadata in Git, never as CSV bytes.
- Required columns, in source order: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`.
- Fail clearly on a malformed header, schema drift, or short rows.
- Raw values are immutable: `Churn` stays `Yes/No` and blank `TotalCharges` stays blank.
- Source version is the SHA-256 of the file bytes, recorded in every manifest.

### Database

- Schemas: `raw`, `staging`, `features`, `predictions`.
- `raw.telco_customers` stores every source field as `TEXT`, including `SeniorCitizen` and `TotalCharges`, plus a `source_version` column. It is append-only by version.
- `raw.ingestion_log` has `source_version` as primary key. This is the idempotency key: a version already logged is never loaded again.
- `staging.telco_customers` uses explicit types and primary key `(source_version, customerid)`. Column names are lowercase source names.
- All reads by downstream stages select the source version named in `ingest_manifest.json`. No stage reads "latest" or unversioned rows.
- Staging must contain exactly as many rows as the source. A count mismatch (for example from duplicate IDs) aborts the transaction.
- Unknown target labels, missing or duplicate IDs, invalid numerics, or schema drift fail with an actionable error and leave no partial data.
- SQL uses `CREATE ... IF NOT EXISTS` and never `DROP`, `TRUNCATE`, or bulk `DELETE`.

### Staged-data contract (Pandera, strict)

- `customerid`: string, non-empty, unique.
- `seniorcitizen`: integer in {0, 1}. `tenure`: integer >= 0. `monthlycharges`: float >= 0.
- `totalcharges`: float >= 0, nullable. Nulls are allowed and never imputed in this phase.
- `churn`: integer in {0, 1}.
- Every categorical field is restricted to its known category set. Extra or missing columns fail.
- Failures are summarized by column and check only, bounded to 50 groups, with no row payloads.

### Split artifacts

- Ratios (0.70 / 0.15 / 0.15), seed 42, stratified by `churn`.
- Two-stage stratified split on rows sorted by `customerid`, so assignment is deterministic.
- Assert ratios sum to one, partitions are disjoint, and together they cover every staged row.
- Partitions are written to `data/interim/{train,validation,test}.csv` as DVC outputs. They may contain IDs for overlap checks but never feed reports, plots, logs, or feature lists.
- `split_manifest.json` records source version, seed, ratios, partition sizes, and per-split churn rate. The churn rate is a stratification integrity check only.
- Phase 4 must reuse these artifacts, not recreate them.

### Analysis artifacts

- `structure_profile.json` (full dataset, target unused): row and column counts, dtypes, missing counts, unique-ID count.
- `analysis_metrics.json` (training partition only): churn rate, numeric summaries and correlation with `churn`, categorical level counts and churn rates, Cramér's V, sparse levels (fewer than 30 rows), and a count of blank `TotalCharges` rows with `tenure = 0`.
- Proxy or leakage review flags: Cramér's V above 0.5 or absolute correlation above 0.8. Flags are for Phase 3 review and never remove or select a feature automatically.
- `docs/data-analysis-report.md` is rendered from the metrics by code. It states the source version, the generating command, the scope of each statistic, and limitations (exploratory, non-causal, multiple comparisons, sparse categories).
- No report may contain raw rows, `customerID` values, secrets, hand-written metrics, or model-quality claims.

## Implementation plan

1. **Environment contract.** Update `.env.example` with names only: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, MLflow URI, and `ANTHROPIC_API_KEY`. No values. Missing database variables stop the run with a clear message that never prints values.
2. **DVC and source placement.** Run `dvc init` without a remote. Copy the source only if the destination is absent, verify checksum and row count, then `dvc add`. Commit only DVC metadata and ignore rules.
3. **PostgreSQL bootstrap.** Add a pinned service, health check, named volume, and `${VAR:?}` wiring so no credential is committed. Bind the port to `127.0.0.1`.
4. **Ingestion.** In one transaction: insert the ingestion-log row, load raw rows only if that row was new, apply the staging insert (`ON CONFLICT DO NOTHING`), and verify the staged row count. Log counts and version prefixes only.
5. **Validation.** Load the staged frame for the manifest's source version, run Pandera with `lazy=True`, write the bounded validation report and `structure_profile.json`, and exit non-zero on failure.
6. **Splitting.** Refuse to run unless the validation report says `passed`. Write partitions and the split manifest only after all assertions pass.
7. **Analysis.** Read only `structure_profile.json` and `train.csv`. The stage must not read the database, validation, or test data.
8. **Entry point and docs.** Production logic lives in `src/churn_phase2.py`, one function per stage, run as `python src/churn_phase2.py {bootstrap|ingest|validate|split|analyze|all}`. Splitting it into a `src/data/` package is allowed later but not required here. README documents commands, variable names, artifact paths, rerun behavior, and teardown cautions.

## Pipeline and lineage

`dvc.yaml` declares four stages:

1. `ingest`: depends on the raw CSV and `src/churn_phase2.py`; outputs `ingest_manifest.json` (source version, row count, staged row count; no run-specific fields, so reruns stay byte-stable).
2. `validate`: depends on the ingest manifest and code; outputs `validation_report.json` and `structure_profile.json`.
3. `split`: depends on the validation report and code; outputs the three partitions and `split_manifest.json`.
4. `analyze`: depends on `train.csv`, `structure_profile.json`, and code; outputs `analysis_metrics.json` and the report. Validation and test partitions are never dependencies.

Rules:

- Database state is not a DVC output, so the ingest manifest is the verification record for it.
- If the PostgreSQL volume is wiped, `dvc repro` will not notice. Document `dvc repro -f ingest` for that case.
- Commit `dvc.lock`. Check ordering with `dvc dag` and reproducibility with two consecutive `dvc repro` runs. The second must report no changed work.
- Do not configure or push a DVC remote.

## Files to create

- `.dvc/config` and DVC metadata (no remote, no credentials).
- `data/raw/.gitignore` and `data/raw/Telco-Customer-Churn.csv.dvc`.
- `docker-compose.yml`: PostgreSQL service, health check, named volume.
- `src/churn_phase2.py`: bootstrap, ingest, validate, split, analyze, and deterministic report rendering.
- `dvc.yaml`, and `dvc.lock` after the first `dvc repro`.
- `docs/data-analysis-report.md`: generated by the pipeline, never hand-edited.
- `tests/data/`: unit tests for staging, schema, split, analysis, and privacy.
- `tests/integration/test_postgres_ingestion.py`: Docker-backed ingestion and idempotency test, marked `integration`.

Generated customer-bearing outputs under `data/interim/` and `data/processed/` stay out of Git and are tracked through DVC.

## Files to modify

- `pyproject.toml`: add dependencies and the `integration` pytest marker; keep existing tool settings.
- `.env.example`: names only; rename the generic LLM key to `ANTHROPIC_API_KEY`.
- `.gitignore`: keep data and secret protection, allow DVC metadata.
- `README.md`: Phase 2 commands and artifact locations, with no unexecuted metrics.

No migration files exist and none may be edited.

## Dependencies

Add to `pyproject.toml` with Python 3.11 bounds:

- `pandas`: staging frames and aggregate analysis.
- `pandera[pandas]`: executable staged-data contract.
- `sqlalchemy`: explicit transactional database access.
- `psycopg[binary]`: PostgreSQL driver.
- `scikit-learn`: the stratified split only.
- `scipy`: chi-square for Cramér's V. Declare it directly rather than relying on the transitive install.
- `dvc`: local lineage.

No plotting libraries: tables carry the evidence. Do not add Prefect, MLflow, SHAP, XGBoost, PyTorch, Optuna, Anthropic, FAISS, FastAPI, Streamlit, or Evidently. Explain each dependency in the completion report.

## Tests and validation

### Unit and contract tests

- The CSV loads with the exact source columns and row count, without mutation or printed records.
- Ingestion is idempotent per source version and never updates raw values.
- Staging maps only `Churn` Yes/No to 1/0, turns blank `TotalCharges` into NULL, and keeps every row.
- Pandera rejects each of these: missing columns, invalid target values, duplicate or empty IDs, wrong types, and out-of-contract categories.
- Null `TotalCharges` passes validation and stays unimputed.
- The split is deterministic at seed 42, stratified within integer rounding, exhaustive, and disjoint by ID.
- `Churn` and `customerID` appear in no declared feature list, and no learned transformer is fitted.
- `analyze` reads only the training partition and the structure profile.
- Reports and logs contain no IDs, raw rows, secrets, or `.env` values.
- Repeated analysis produces byte-identical output for identical inputs.

### Failure-path tests (required)

- A duplicate customer ID aborts ingestion and leaves the ingestion log unchanged.
- An unknown `Churn` label aborts ingestion and leaves no partial rows.
- A bad category passes ingestion, fails validation with exit code 1, and `split` then refuses to run.

### Integration and lineage checks

- `docker compose config` resolves with placeholder values.
- PostgreSQL reports healthy and the four schemas exist.
- A full load produces raw and staged row counts equal to the source.
- A second load changes nothing.
- `dvc dag` shows `ingest -> validate -> split -> analyze`.
- `dvc repro` succeeds twice, and the second run reports no changed work.
- `git status --short` never lists raw, interim, or processed customer data.

### Required completion commands

Run from the repository root and report the actual exit status and output:

```sh
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
docker compose config
docker compose up -d postgres
.venv/bin/pytest -m integration
.venv/bin/dvc dag
.venv/bin/dvc repro
```

If Docker is unavailable, report integration checks as unavailable, not passed. If pytest collects no relevant tests, report that as inconclusive. Do not skip or broadly catch failures to get a green result.

## Safety and approval gates

- Raw data is immutable, excluded from Git, and tracked by DVC metadata only.
- Never overwrite the canonical raw path, delete the original source, rewrite DVC or MLflow artifacts, or edit an applied migration.
- Map `Churn` to 1/0 only in staging.
- Keep blank `TotalCharges` as NULL until imputation inside the future training pipeline. Never drop rows silently.
- Exclude `Churn` and `customerID` from model features and from all LLM inputs and logs.
- Split before any fitting, using the configured seed.
- Fit learned preprocessing on training data only, in the future shared pipeline.
- Keep validation and test out of exploratory decisions.
- Never read or print `.env` values. This phase makes no external model or API calls.
- No `DROP`, `TRUNCATE`, bulk `DELETE`, destructive teardown, or volume removal without explicit approval.
- No DVC remote, push, deployment, paid resource, threshold change, or model promotion without explicit approval.

## Definition of done

- [ ] The source is byte-identical at the canonical raw path, protected from Git, and represented by DVC metadata. The original file is not deleted.
- [ ] Docker Compose starts a healthy PostgreSQL service with no committed credentials.
- [ ] The four schemas and the raw, staging, and ingestion-log tables are created without destructive SQL.
- [ ] Ingestion and staging are transactional, idempotent, and preserve every source row.
- [ ] Pandera accepts the real staged data and rejects every required known-bad case.
- [ ] Blank `TotalCharges` is NULL and unimputed.
- [ ] The seed-42 70/15/15 split is DVC-versioned, disjoint, exhaustive, stratified, and reproducible.
- [ ] Structure evidence and training-only EDA come from executed code, labeled with source version and scope.
- [ ] No report, log, Git-tracked file, or feature list exposes identifiers, raw rows, secrets, or `Churn` and `customerID` improperly.
- [ ] `dvc dag` and two `dvc repro` runs show the declared order and stable reruns.
- [ ] Failure-path tests pass.
- [ ] PostgreSQL integration tests pass against the real local service.
- [ ] `ruff check .` and `ruff format --check .` exit 0.
- [ ] `pytest` collects the Phase 2 tests and exits 0.
- [ ] The completion report lists changed files, new dependencies with reasons, commands run with actual results, the source version, limitations, and confirms no remote, deployment, promotion, threshold, or paid action.

## Handoff

Phase 3 receives:

- a DVC-pinned immutable raw source and a validated staging contract;
- fixed partitions and a split manifest that must be reused;
- idempotent data-access code and schema;
- aggregate quality evidence and training-only findings;
- documented missingness (including whether blank `TotalCharges` rows all have `tenure = 0`), category, cardinality, and proxy risks.

Phase 3 uses this to implement `database/feature_views.sql` and the one shared scikit-learn preprocessing pipeline. It keeps `Churn` and `customerID` out of features, fits learned transformations on training data only, and adds leakage and train/serve parity tests before Phase 4.

## Revision notes

Changes from the first draft of this spec:

- Full-dataset analysis no longer uses the target. Per-split churn rate appears only as a split-integrity check.
- `analyze` depends on `structure_profile.json` and `train.csv`, not on database state.
- The source version a stage reads is stated explicitly.
- Raw columns are `TEXT` so values are truly unchanged.
- Staging row-count equality is required, because `ON CONFLICT DO NOTHING` can otherwise hide duplicate IDs.
- `scipy` is declared directly.
- A stale-manifest limitation and its `dvc repro -f ingest` workaround are documented.
- Failure-path tests are required.
- Code lives in one module (`src/churn_phase2.py`) instead of a `src/data/` package, and plotting libraries are dropped.