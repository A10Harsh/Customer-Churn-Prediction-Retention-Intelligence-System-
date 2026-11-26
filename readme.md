# Customer Retention Intelligence System

An end-to-end, production-oriented customer retention intelligence system. The
implemented foundation currently provides validated data ingestion, reproducible
development/hold-out splitting, exploratory notebooks, automated development-data
EDA, reproducible statistical reporting, and reusable feature preprocessing.
Modeling and recommendation capabilities remain planned.

## Business Objective

Identify customers at risk of churn, explain their primary churn risk drivers and generate personalized retention recommendations within approved business policies.

## Dataset

IBM Telco Customer Churn dataset.

## Implemented Capabilities

- Schema and business-rule validation with Pandera
- Reproducible stratified development/hold-out split
- DVC pipeline with separate EDA and statistical-analysis branches after `split`
- Fifteen executed development-only analytical notebooks
- Reusable descriptive, inference, multiplicity, multicollinearity, segmentation,
  and outlier analysis modules
- Deterministic JSON, CSV, PNG, and HTML EDA reports
- Deterministic confidence-interval, hypothesis-test, adjusted-p-value, and
  multicollinearity reports
- Atomic report publication and explicit held-out-data access protection
- Automated tests for pipeline correctness, failure handling, and reproducibility
- Six deterministic candidate features and cloneable baseline/engineered
  preprocessing pipelines for linear, neural, and tree model families
- Explicit feature subsets and leakage-safe target separation (`No=0`, `Yes=1`)

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

## Current Technology Stack

Python 3.11, pandas, NumPy, SciPy, scikit-learn, Pandera, PyArrow,
Matplotlib, seaborn, DVC, Pytest, and Ruff.

## Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis]"
python -m pip check
```

The package currently supports Python `>=3.11,<3.12`. `requirements-lock.txt`
records the fully resolved development environment used for reproducibility.

## Pipeline and Reports

Retrieve DVC-managed inputs when a configured remote is available, then reproduce
the pipeline:

```bash
dvc pull
dvc stage list
dvc dag
dvc status
dvc repro --dry
dvc repro
```

Run only the automated EDA stage directly with:

```bash
python -m retention.analysis.eda
```

Run only the automated statistical-analysis stage with:

```bash
python -m retention.analysis.statistical_analysis
```

The stage reads only `data/processed/development.parquet`. Generated artifacts are
written to `reports/generated/eda/`:

- `summary.json`: strict machine-readable report
- `report.html`: executive report with relative artifact links
- `tables/`: analytical CSV tables
- `figures/`: noninteractive PNG figures

See [the automated EDA documentation](docs/eda_pipeline.md) and
[the statistical-analysis documentation](docs/statistical_analysis.md), plus
[the data-pipeline documentation](docs/data_pipeline.md), for definitions,
dependencies, leakage controls, and limitations. Statistical artifacts are written
to `reports/generated/statistical_analysis/` as strict JSON, five focused CSV
tables, and a standalone HTML report.

For feature definitions, the six preprocessing configurations, and future
cross-validation usage, see [feature preprocessing](docs/preprocessing.md).

## Development Checks

```bash
python -m pytest -v
python -m pytest --cov=retention --cov-report=term-missing
ruff check .
ruff format --check .
```

Executed notebooks are preserved research artifacts and are excluded from Ruff's
automatic lint/format scope. Production Python modules and tests remain fully
checked. Notebook code quality is reviewed during notebook development without
rewriting completed artifacts.

## Data-Leakage Boundary

- Automated EDA may read only the development split.
- Automated statistical analysis may read only the development split.
- The held-out split is reserved for later final model evaluation.
- EDA associations are descriptive and do not perform feature selection.
- Statistical tests are EDA-informed exploratory evidence, not independent
  confirmation or causal effects.
- Feature engineering is customer-local. Imputation, scaling, and categorical
  mappings are learned only when a preprocessing pipeline is fitted; future
  model evaluation must fit it inside each training fold.
- No globally preprocessed dataset, class balancing, or supervised model
  training has been implemented.

## Project Status

- [x] Phase 1: Project initialization
- [x] Phase 2: Data ingestion and validation
- [x] Phase 3A: Development-data EDA and automated reporting
- [x] Phase 3B: Statistical hypothesis testing and automated reporting
- [x] Phase 3C: Feature engineering and preprocessing readiness
- [ ] Phase 4: Machine learning
- [ ] Phase 5: Deep learning
- [ ] Phase 6: Experimentation and model selection

## Disclaimer

This project is an educational implementation using historical customer data.
Observed churn associations and monthly-charge exposure are descriptive. They do
not establish causal effects, realized revenue loss, customer lifetime value, or
retention-treatment effectiveness.
