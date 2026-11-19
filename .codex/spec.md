# Churn Retention System Specification

## 1. Objective

Build a production-style customer retention system that:

- predicts customer churn;
- explains churn drivers with SHAP;
- recommends guarded retention actions with an LLM;
- serves results through FastAPI;
- monitors deployed behavior with Evidently.

## 2. Core stack

- Python 3.11, PostgreSQL, and Docker Compose
- Pandera, DVC, scikit-learn, XGBoost, PyTorch, and Optuna
- MLflow, SHAP, Anthropic API, FAISS, and Pydantic
- FastAPI, Prefect, Streamlit, Evidently, and GitHub Actions

Use logistic regression as the required baseline. Compare Random Forest, XGBoost, and a direct PyTorch ANN through a serializable scikit-learn-compatible adapter. Never use `MLPClassifier`; every family is serving-eligible after passing the same evaluation, approval, schema, latency, and rollback gates.

## 3. Repository rules

- Put production logic in `src/`; notebooks are exploratory only.
- Keep non-secret settings in `configs/*.yaml` and secrets in the environment.
- Add dependencies to `pyproject.toml` in the change that uses them.
- Type all functions and methods; document public functions.
- Ask before creating top-level folders.
- Do not refactor unrelated files.

## 4. Data contract

- Source: IBM Telco Customer Churn CSV in `data/raw/`, tracked with DVC.
- Identifier: `customerID`; target: `Churn`.
- Map `Yes/No` to `1/0` only in staging.
- Never modify raw records or commit customer data to Git.
- Coerce blank `TotalCharges` to missing numeric values.
- Impute missing values inside the training pipeline.
- Never drop rows silently.
- Measure dimensions and class balance from the actual file.

## 5. Database contract

- PostgreSQL schemas: `raw`, `staging`, `features`, and `predictions`.
- Define base schemas in `database/schema.sql`.
- Make schema changes through new numbered migrations.
- Never edit applied migrations or run destructive SQL without approval.

## 6. Validation

- Define staged-data contracts with Pandera.
- Validate required columns, types, target values, and identifier behavior.
- Permit expected missing `TotalCharges` values.
- Test that known-bad staged data fails clearly.

## 7. Feature pipeline

- Build one shared scikit-learn pipeline for training, scoring, and serving.
- Exclude `Churn` and `customerID` from features.
- Prevent target, temporal, and proxy leakage.
- Fit imputers and encoders on training data only.
- Handle unseen categories and deterministic feature order.
- Test train/serve feature parity.

## 8. Data split

- Split before fitting anything.
- Use stratified `70/15/15` train/validation/test partitions.
- Use configured seed `42` and ensure no overlap.
- Tune and cross-validate with training data only.
- Calibrate and select the threshold with validation data only.
- Evaluate test data only for final candidates.
- Never use test results for tuning or selection.

## 9. Model development

- Log an untuned logistic-regression baseline before all challenger experiments.
- Run comparable Random Forest, XGBoost, and PyTorch ANN experiments on identical splits and folds.
- Tune shortlisted challengers with bounded Optuna cross-validation on training data.
- Select models by validation PR-AUC.
- Calibrate probabilities with Platt or isotonic calibration.
- Select the threshold using configured business costs.
- Ask before changing an approved threshold.
- Report real PR-AUC, ROC-AUC, precision, recall, and F1 results.
- Report Brier score or a calibration curve.
- Never fabricate or estimate metrics.

## 10. MLflow contract

- Use experiment `churn-retention` and model name `churn-classifier`.
- Log seed, split configuration, and DVC data version.
- Log parameters, metrics, calibration, and threshold evidence.
- Log the serialized pipeline and model artifacts.
- Ask before champion promotion.

## 11. Explainability

- Generate global and per-customer SHAP explanations.
- Use the approved model artifact and readable feature names.
- Exclude identifiers from explanations.
- Test output shape, finite values, and determinism.

## 12. GenAI recommendations

- Retrieve approved retention playbook content with FAISS.
- Read the Anthropic model from configuration and key from the environment.
- Allow only probability, SHAP drivers, segment, contract type, and playbook text.
- Never send names, contact details, raw records, or `customerID`.
- Parse every response with Pydantic.
- Enforce configured business guardrails.
- Return a safe default action on failure.
- Log only non-sensitive reason codes.
- Use a fake LLM client in tests and CI.

## 13. API contract

- Provide `POST /predict`, `/batch-predict`, and `/recommend`.
- Validate every request and response.
- Return calibrated probabilities from the training pipeline.
- Write auditable results to the `predictions` schema.
- Do not expose secrets, PII, stack traces, or raw identifiers.
- Test valid and invalid payloads for every endpoint.

## 14. Orchestration and monitoring

- Create Prefect flows for ingest, validate, train, score, and monitor.
- Make flows explicit, repeatable, and retry-safe.
- Generate Evidently reports from real reference and current data.
- Monitor schema, missingness, feature drift, and performance.
- Record data windows, report time, and model version.
- Do not retrain, promote, or deploy automatically.

## 15. Dashboard

- Show measured risk segments and probability distributions.
- Show retention opportunity, SHAP drivers, and Next-Best-Actions.
- Show monitoring status and artifact versions.
- Handle loading, empty, stale, and error states.
- Do not expose secrets or unnecessary identifiers.

## 16. Delivery order

Complete one phase at a time:

1. Scaffold — complete
2. Database and ingestion
3. Features
4. Training
5. Explainability
6. GenAI NBA
7. Serving
8. Orchestration and monitoring
9. Dashboard
10. CI/CD and documentation

Do not begin later work before current acceptance checks pass.

## 17. Completion gate

- Run `.venv/bin/ruff check .`.
- Run `.venv/bin/ruff format --check .`.
- Run `.venv/bin/pytest`.
- Run relevant integration or smoke checks.
- Report actual outcomes, changed files, and limitations.
- Treat no collected tests as inconclusive, not passed.
- Never hide, suppress, or broadly catch failures.

## 18. Approval gates

Explicit approval is required before:

- changing the selected threshold or promoting a model;
- editing an applied migration;
- deleting protected data or artifacts;
- running destructive database operations;
- pushing to a remote or deploying;
- creating paid resources or external costs.
