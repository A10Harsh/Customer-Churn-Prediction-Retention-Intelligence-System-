# Customer Retention Intelligence System

An end-to-end, production-oriented customer retention intelligence system that combines machine learning, deep learning, explainable AI and generative AI to support customer retention decisions.

## Business Objective

Identify customers at risk of churn, explain their primary churn risk drivers and generate personalized retention recommendations within approved business policies.

## Dataset

IBM Telco Customer Churn dataset.

## Planned Capabilities

- Customer churn probability estimation
- Customer risk segmentation
- Machine learning and deep learning model benchmarking
- Customer-level SHAP explanations
- LLM-powered Next-Best-Action recommendations
- Experiment tracking and model versioning
- REST API and interactive analytics dashboard
- Data drift and model performance monitoring
- Automated testing and CI/CD

## Technology Stack

Python, Scikit-learn, XGBoost, PyTorch, Optuna, Pandera, DVC, MLflow, PostgreSQL, SQLAlchemy, Alembic, SHAP, FastAPI, Streamlit, Plotly, Docker, GitHub Actions and Evidently.

## Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Development Checks

```bash
pytest
ruff check .
ruff format --check .
```

## Project Status

Under active development.

Phase 1: Repository initialization and development environment setup.

## Disclaimer

This project is an educational implementation using historical customer data. Retention recommendations are decision-support outputs and are not evidence of actual treatment effectiveness.