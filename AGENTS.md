# Agent instructions

## Project goal

Build a production-style customer churn retention system that predicts churn, explains drivers with SHAP, recommends retention actions with an LLM, serves predictions and recommendations through FastAPI, and monitors deployed behavior with Evidently.


## Stack

- Python 3.11; PostgreSQL in Docker with `raw`, `staging`, `features`, and `predictions` schemas.
- Pandera for data validation; DVC for dataset and model artifact versioning.
- scikit-learn, XGBoost (primary model), and Optuna for modeling and tuning.
- PyTorch is optional: use it only for a small comparison MLP in Phase 4, and only after the XGBoost baseline is logged. Do not add it to the serving path.
- MLflow for experiment tracking and model registry; SHAP for explanations.
- GenAI Next-Best-Action: Anthropic API with RAG (FAISS) over a retention playbook; Pydantic for schema-validated structured output.
- FastAPI for serving; Prefect for orchestration; Streamlit for the dashboard.
- Evidently for monitoring; Docker and docker-compose for local environments; GitHub Actions for CI/CD.

## Folder architecture

```text
churn-retention-system/
├── AGENTS.md
├── .codex/                   # agent config; read-only unless   asked
├── pyproject.toml
├── .env.example              # required variable names only, no values
├── docker-compose.yml        # postgres, mlflow, api, dashboard
├── configs/                  # non-secret *.yaml (paths, model params, thresholds)
├── data/                     # DVC-tracked; raw/, interim/, processed/ (never in git)
├── database/
│   ├── schema.sql            # raw, staging, features, predictions schemas
│   ├── feature_views.sql     # window functions, CTEs, cohort features
│   └── migrations/           # numbered SQL files; never edit applied ones
├── pipelines/                # Prefect flows: ingest, validate, train, score, monitor
├── src/
│   ├── data/                 # loading and Pandera schemas
│   ├── features/             # feature pipeline shared by training and serving
│   ├── models/               # training, tuning, calibration, evaluation
│   ├── explain/              # SHAP global and per-customer explanations
│   ├── genai/                # RAG retrieval, prompts, output schemas, guardrails, evals
│   ├── api/                  # FastAPI app: /predict, /batch-predict, /recommend
│   └── utils/                # logging, config loading, DB helpers
├── frontend/streamlit/       # KPIs, risk segments, retention opportunity, NBA viewer
├── monitoring/               # Evidently configs and generated reports
├── deployment/               # Dockerfiles and cloud configs
├── tests/                    # mirrors src/ layout; unit and API tests
├── notebooks/                # EDA only; no production logic lives here
├── docs/                     # model card, validation report, retention playbook
└── .github/workflows/        # CI: ruff, pytest, docker build
```

Put new code in the folder that matches its purpose. Ask before creating new top-level folders.

## Dataset

- Source: IBM Telco Customer Churn (Kaggle). Place the raw CSV at `data/raw/` and track it with DVC.
- Target: `Churn` (Yes/No), mapped to 1/0 in staging, never in `raw`.
- Identifier: `customerID`. Exclude from model features and from all LLM prompts and logs.
- Known quirk: `TotalCharges` contains blank strings for customers with zero tenure. Coerce to numeric in staging and handle the missing values inside the training Pipeline imputer. Do not drop rows silently.
- Verify row count, column count, and class balance by running code against the file. Do not take them from this document or from memory.

## Split and evaluation protocol

- Split once, before any fitting: stratified train/validation/test at 70/15/15, seed `42`,  defined in `configs/*.yaml`.
- Fit scalers, encoders, and imputers on the training split only, inside a scikit-learn Pipeline.
- Tune on train/validation only (Optuna with cross-validation on train). The test set is evaluated once per candidate final model and never used for tuning.
- Model selection metric: PR-AUC on validation. Also report ROC-AUC, precision, recall, F1, and Brier score or a calibration curve.
- Calibrate probabilities (isotonic or Platt) on validation data, not on test.
- Choose the decision threshold on validation using a cost-based rule from `configs/` (cost of a missed churner vs cost of an unnecessary offer). Log the comparison. Changing the threshold requires asking first.
- Log every run to MLflow with seed, data version (DVC hash), parameters, and metrics.
- MLflow experiment name: `churn-retention`. Registered model name: `churn-xgb`. Champion promotion requires asking first.

## GenAI Next-Best-Action

- Provider: Anthropic API. Key in `ANTHROPIC_API_KEY` (name only in `.env.example`). Model name comes from `configs/`, not code.
- Retrieval: FAISS index over the retention playbook in `docs/` or `data/`, built by a script in `src/genai/`.
- Inputs to the LLM are limited to churn probability, top SHAP drivers, segment, and contract type.
- Output must parse into a Pydantic schema, then pass business-rule guardrails (for example, maximum discount percentage from config). On failure, return the default action and log the reason.
- Tests and CI use a fake LLM client. CI never makes real API calls and never needs a real key.

## Phase roadmap

Work on one phase at a time. A phase is done only when its criteria are met and the checks have actually been run.

1. **Scaffold** (done): repository, config, hooks.
2. **Database and ingestion**: Docker Postgres up; `database/schema.sql` creates the four schemas; raw CSV loaded to `raw`; staging transform; Pandera schema validates staging; tests pass.
3. **Features**: `database/feature_views.sql` plus a `src/features/` sklearn pipeline used identically by training and serving; leakage and train/serve parity tests pass.
4. **Training**: stratified split, XGBoost baseline, Optuna tuning, calibration, threshold selection, MLflow logging; optional PyTorch MLP comparison. Metrics come only from logged runs.
5. **Explainability**: SHAP global summary and per-customer top drivers; outputs saved and tested for shape and determinism.
6. **GenAI NBA**: playbook, FAISS retrieval, prompts, Pydantic schema, guardrails, fake-client tests, small eval set.
7. **Serving**: FastAPI `/predict`, `/batch-predict`, `/recommend` with request/response schema tests; predictions written to the `predictions` schema.
8. **Orchestration and monitoring**: Prefect flows for ingest, validate, train, score, monitor; Evidently drift and performance reports generated from real data.
9. **Dashboard**: Streamlit KPIs, risk segments, retention opportunity, NBA viewer, reading from the API or `predictions` schema.
10. **CI/CD and docs**: GitHub Actions (ruff, pytest, docker build), model card, architecture diagram, validation report, README with only measured numbers.

## Conventions

- Add type hints to all functions and methods. Keep functions small and add docstrings to public functions.
- Use Ruff for linting and formatting, and pytest for tests.
- Store non-secret configuration in `configs/*.yaml`. Do not hardcode paths or credentials.
- Keep secrets only in local `.env` files. Never commit them; document required names in `.env.example`, including database host, port, name, and user.
- Set and log random seeds so every training run is reproducible.
- Production logic lives in `src/`; notebooks only call into it for exploration.
- Any new dependency must be added to `pyproject.toml` in the same change.

## Testing expectations

- Leakage test: `Churn` and `customerID` are absent from the model feature set.
- Train/serve parity test: the same raw record produces identical features in training and serving.
- API tests: valid and invalid payloads for each endpoint, checked against the response schema.
- Guardrail test: invalid or out-of-policy LLM output falls back to the default action.
- Pandera test: known-bad staging data fails validation.
- Early phases with no testable code yet may have an empty test suite; say so explicitly rather than reporting a pass. `pytest` exiting with "no tests collected" is reported as inconclusive, not as success.

## Rules for the agent

- Never commit data or secrets.
- Run Ruff and pytest before declaring a task done; report any unavailable or inconclusive check.
- Explain each new dependency before adding it.
- Do not refactor files outside the current task.
- Work on one phase or task at a time, and list the files you changed when you finish.
- Run the code and report real outputs. Do not describe results you have not produced.
- Follow `.codex/spec.md` for requirements. If a task conflicts with it, ask before proceeding.

## Warnings (read before every task)

- **Never fabricate metrics.** ROC-AUC, precision, recall, KPIs, and drift numbers must come from an actual run logged in MLflow or a generated report. If a run fails or is not executed, say so. Do not fill in placeholder numbers anywhere, including the README and resume text.
- **Prevent data leakage.** Split train/validation/test before any fitting. Fit scalers, encoders, and imputers on the training split only, inside a scikit-learn Pipeline. Exclude the target (`Churn`) and identifiers (`customerID`) from model features. Do not tune on the test set.
- **No PII to the LLM.** Send only structured, non-identifying inputs (churn probability, top SHAP drivers, segment, contract type). Never send names, contact details, or raw customer IDs in prompts or logs.
- **Validate and guard LLM output.** Parse every recommendation through a Pydantic schema and apply business-rule guardrails (for example, a maximum discount). Reject or fall back to a default action when validation fails.
- **Protect the database.** Do not run `DROP`, `TRUNCATE`, or bulk `DELETE` without explicit confirmation. Never modify `raw` tables; transform into `staging` and `features`. Schema changes go in new numbered migration files.
- **Protect artifacts.** Do not overwrite or delete DVC-tracked data, MLflow runs, or registered model versions. Do not promote a model to champion or change the decision threshold without logging the comparison and asking first.
- **No external side effects.** Do not push to remote, deploy, create cloud resources, or run commands that incur cost unless asked.
- **Class imbalance.** Report precision, recall, and PR-AUC alongside ROC-AUC, and use calibrated probabilities for threshold decisions. Accuracy alone is not acceptable.
- **No silent failures.** Do not wrap errors in broad `try/except` or skip failing tests to get a green result. Surface the failure.

## Commands

Run from the repository root with Python 3.11:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
.venv/bin/pre-commit run --all-files
tree -L 3
```

## Current status

- Phase 1: repository scaffold, project configuration, and development hooks created.
- Next: Phase 2 (database and ingestion), following `.codex/spec.md`.
- Data ingestion, validation, feature engineering, training, serving, monitoring, and deployment remain to be implemented.