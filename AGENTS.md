# Agent instructions

## Project goal

Build a production-style customer churn retention system that predicts churn, explains drivers with SHAP, recommends retention actions with an LLM, serves predictions and recommendations through FastAPI, and monitors deployed behavior with Evidently.

## Stack

- Python 3.11; PostgreSQL in Docker with `raw`, `staging`, `features`, and `predictions` schemas.
- Pandera for data validation; DVC for dataset and model artifact versioning.
- scikit-learn, XGBoost or LightGBM, PyTorch, and Optuna for modeling and tuning.
- MLflow for experiment tracking and model registry; SHAP for explanations.
- FastAPI for serving; Prefect for orchestration; Streamlit for the dashboard.
- Evidently for monitoring; GitHub Actions for CI/CD.

## Conventions

- Add type hints to all functions and methods. Keep functions small and add docstrings to public functions.
- Use Ruff for linting and formatting, and pytest for tests.
- Store non-secret configuration in `configs/*.yaml`. Do not hardcode paths or credentials.
- Keep secrets only in local `.env` files. Never commit them; document required names in `.env.example`.

## Rules for the agent

- Never commit data or secrets.
- Run Ruff and pytest before declaring a task done; report any unavailable or inconclusive check.
- Explain each new dependency before adding it.
- Do not refactor files outside the current task.

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
