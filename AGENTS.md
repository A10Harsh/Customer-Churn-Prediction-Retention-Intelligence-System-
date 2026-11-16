# Agent instructions

## Project goal

Build a production-style customer churn retention system that predicts churn, explains drivers with SHAP, recommends retention actions with an LLM, serves predictions and recommendations through FastAPI, and monitors deployed behavior with Evidently.

## Stack

- Python 3.11; PostgreSQL in Docker with `raw`, `staging`, `features`, and `predictions` schemas.
- Pandera for data validation; DVC for dataset and model artifact versioning.
- scikit-learn, XGBoost or LightGBM, PyTorch, and Optuna for modeling and tuning.
- MLflow for experiment tracking and model registry; SHAP for explanations.
- GenAI Next-Best-Action: an LLM API with RAG (pgvector or FAISS) over a retention playbook; Pydantic for schema-validated structured output.
- FastAPI for serving; Prefect for orchestration; Streamlit for the dashboard.
- Evidently for monitoring; Docker and docker-compose for local environments; GitHub Actions for CI/CD.

## Folder architecture

```text
churn-retention-system/
├── AGENTS.md
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
├── docs/                     # architecture diagram, model card, validation report
└── .github/workflows/        # CI: ruff, pytest, docker build
```

Put new code in the folder that matches its purpose. Ask before creating new top-level folders.

## Conventions

- Add type hints to all functions and methods. Keep functions small and add docstrings to public functions.
- Use Ruff for linting and formatting, and pytest for tests.
- Store non-secret configuration in `configs/*.yaml`. Do not hardcode paths or credentials.
- Keep secrets only in local `.env` files. Never commit them; document required names in `.env.example`.
- Set and log random seeds so every training run is reproducible.
- Production logic lives in `src/`; notebooks only call into it for exploration.

## Rules for the agent

- Never commit data or secrets.
- Run Ruff and pytest before declaring a task done; report any unavailable or inconclusive check.
- Explain each new dependency before adding it.
- Do not refactor files outside the current task.
- Work on one phase or task at a time, and list the files you changed when you finish.
- Run the code and report real outputs. Do not describe results you have not produced.

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
- Data ingestion, validation, feature engineering, training, serving, monitoring, and deployment remain to be implemented.
