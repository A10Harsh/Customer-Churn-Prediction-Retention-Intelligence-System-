# Production explanation contract (Phase 8)

## Model and output space

The trusted `artifacts/final_model/predictor.joblib` supplies the fitted
preprocessor, XGBoost estimator, sigmoid calibrator, and frozen decision
threshold. `src/retention/explainability/service.py` initializes one cached
`shap.TreeExplainer` per process. It explicitly uses
`feature_perturbation="tree_path_dependent"` and `model_output="raw"`. This
explains the **raw XGBoost log-odds margin**. The tree's stored training-path
counts provide the background distribution, so no held-out or separate
background dataset is read. The choice does not model feature dependence
causally; correlated inputs can share or shift attribution.

For each customer, the service checks that the SHAP base value plus all 45
transformed contributions equals the raw estimator margin within `1e-4`. It
checks the margin's sigmoid against XGBoost's raw probability. The existing
sigmoid calibrator then converts that raw probability to the calibrated churn
probability; the frozen threshold determines the class. SHAP contributions
**are not calibrated probability changes**. This explains the base estimator
behind a calibrated prediction, not the calibration layer itself.

`ColumnTransformer.output_indices_`, fitted transformer column lists, and the
fitted one-hot encoder's category counts map all transformed columns to the 19
original predictors. Signed contributions are summed per original predictor.
This includes reference-coded categories without parsing feature-name strings.
The production contract rejects unsupported or unknown categories before
explanation; the fitted tree path has no dropped reference category. Numeric
charges remain decimal strings in snapshots and JSON output.

The typed `CustomerExplanation` includes the full transformed and original
vectors, absolute ranking, positive and negative top factors, raw and
calibrated outputs, provenance, timestamp, and additivity residual. Positive
factors increased the model's **raw score** for this customer; negative factors
reduced it. Zero contributions do not fill the top lists. Sorting breaks ties
by feature name. `attribute_group` distinguishes immutable household or
demographic characteristics, historical context, and service or billing
attributes. It is descriptive metadata, not a retention recommendation.

## Persistence and migration

Revision `0002_explanations` adds an `explanations` table. Each row references
an immutable prediction with `ON DELETE RESTRICT`. It stores raw-margin
values, original contributions, ranked factors, a full JSONB contract, model
and explainer fingerprints, and a creation timestamp. The unique key combines
prediction, method, version, explainer fingerprint, and `top_k`: identical
requests return the same row, while later explanation versions remain possible.
The orchestration function reads the prediction's saved predictor snapshot,
not current customer attributes. It rejects a different model name, version,
fingerprint, threshold, class, or calibrated probability. A database failure
rolls back and is never reported as successful persistence.

Use `DATABASE_URL` and `TEST_DATABASE_URL` from a private local `.env`. The
existing test fixture upgrades and downgrades only an empty `_test` database;
never downgrade the application database. The local PostgreSQL cluster may use
`trust` authentication on loopback and Unix sockets. Review `pg_hba.conf`
before deployment; Phase 8 does not change server security settings or expose
PostgreSQL externally.

```sh
set -a; . ./.env; set +a
.venv/bin/alembic upgrade head
.venv/bin/alembic current
.venv/bin/alembic check
scripts/test_suite.sh
```

To explain an existing prediction without creating another prediction:

```python
from retention.database.connection import create_database_engine, make_session_factory
from retention.database.explanations import explain_prediction

engine = create_database_engine()
try:
    factory = make_session_factory(engine)
    explanation = explain_prediction(factory, "existing-prediction-uuid")
    print(explanation.to_dict())
finally:
    engine.dispose()
```

The output has no customer identifier inside feature contributions. Do not
log raw customer records or database URLs.

## Global development sample

Run `.venv/bin/python -m retention.explainability.global_explanations` from the
repository root. It reads only the 19 predictor columns of the development
Parquet file, verifies its frozen fingerprint, and samples up to 1,000 rows
with seed `42`. It generates only five charts under
`reports/generated/explainability/`: global mean-absolute importance,
beeswarm, and dependence plots for tenure, Contract, and InternetService.
`summary.json` records mean absolute, mean signed, and distribution quantiles
for every original predictor. Matching model, explainer, development, and
sample fingerprints allow safe reuse. Plots and summary are derived reports,
ignored by Git. Categorical beeswarm values appear grey; use the labeled
dependence plots to inspect category direction.

This sample describes model behavior on development customers. It is not a
new evaluation, intervention effect, causal finding, or production latency
benchmark. Correlated features and categorical coding limit one-factor
interpretations. No held-out observation or label enters these artifacts.
