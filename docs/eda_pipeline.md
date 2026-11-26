# Automated EDA pipeline

Phase 3A Step 13 turns the stable, recurring parts of the seven exploratory
notebooks into independently testable Python functions. The notebooks remain the
research record: they contain broad visual exploration, sensitivity experiments,
and narrative reasoning. The automated pipeline deliberately emits a smaller,
repeatable operational report.

## Responsibilities

- `descriptive.py`: dimensions, quality, cardinality, moments, category frequency,
  rare/near-constant screening, target balance, and the majority baseline.
- `association.py`: Pearson, Spearman, Kendall Tau-b, point-biserial correlation,
  corrected Cramér's V, eta, and exploratory marginal mutual information.
- `segmentation.py`: KPI, lifecycle, charge-quartile, contract, service-adoption,
  contract–lifecycle, risk–value, lift, exposure, reconciliation, and Pareto views.
- `outliers.py`: diagnostic IQR, population Z-score, modified Z/MAD, percentile,
  billing, and service-consistency screens. It never changes observations.
- `visualization.py`: a bounded set of noninteractive PNG figures using Agg.
- `reporting.py`: strict JSON, deterministic CSV, and linked HTML persistence.
- `eda.py`: configuration, input guardrails, orchestration, and atomic publishing.

## Definitions and configuration

Pandas sample variance (`ddof=1`), sample skewness, and excess kurtosis preserve
the notebook conventions. Categories below 1% are rare; a dominant value at 95%
is near-constant. Corrected Cramér's V uses the Bergsma finite-sample correction.
Mutual information is a deterministic, marginal exploratory estimator with three
neighbors and the project random seed. No p-values or formal feature selection are
reported.

Lifecycle boundaries are 0–6, 7–12, 13–24, 25–48, and 49–72 completed months.
Monthly-charge groups use development-population quartiles. Adoption counts phone,
multiple-line, internet, and six internet add-on services, producing 0–2, 3–5,
and 6–9 bands. Risk–value cells require 30 customers and compare observed churn
and mean monthly charges with population means. `MonthlyCharges` is only an
exposure/value proxy—not realized loss, CLV, or causal impact.

All thresholds, feature lists, figure settings, and output paths live under `eda`
in `params.yaml`; the shared `project.random_state` is reused.

## Execution and artifacts

Run `python -m retention.analysis.eda`, or `dvc repro eda`. Output is atomically
published to `reports/generated/eda`: `summary.json`, `report.html`, CSV files in
`tables/`, and PNG files in `figures/`. JSON is strict (`NaN`/infinity become
`null`) and excludes volatile timestamps.

The DVC graph is `ingest -> validate -> split -> eda`. The final stage depends on
`development.parquet`, analysis modules, shared configuration/logging helpers,
the dependency snapshot, EDA parameters, and the project seed. It does not depend
on `test.parquet`, `validated.parquet`, or `split_report.json`.

## Leakage prevention and reproducibility

The orchestrator accepts only the configured file named `development.parquet`,
rejects configured test/validated paths, excludes `customerID` from associations,
and never writes data inputs. Artifact content has no timestamp; deterministic
sorting and serialization support hash comparison. Publishing occurs from a
sibling staging directory, so a failed build cannot replace the last complete
report.

## Limitations and notebook-only work

Results describe one development split and are not hold-out estimates. Association
does not imply causation. Sparse cells are retained and marked; no intervention
effectiveness is estimated. Isolation Forest and LOF remain notebook-only because
their contamination and neighborhood choices are sensitivity experiments rather
than stable recurring data-quality rules. Broad plots, VIF, formal hypothesis
tests, feature engineering, selection, balancing, preprocessing, and supervised
models are outside this phase.

## Repository quality policy

Ruff checks and formats production Python modules and tests. The seven executed
EDA notebooks are research artifacts with verified SHA-256 hashes, so
`notebooks/*.ipynb` is excluded from Ruff's automatic lint and formatting scope.
Notebook code is reviewed when authored, but completed notebook JSON is not
rewritten during repository-wide quality checks.

Local DVC remotes belong in ignored `.dvc/config.local`; shared DVC configuration
must not contain workstation-specific absolute paths. In restricted environments,
DVC's cache location can be redirected temporarily with `XDG_CACHE_HOME` and
`XDG_CONFIG_HOME`. Such workarounds are environment settings and are not committed
to project configuration.
