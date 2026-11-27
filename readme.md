# Customer Retention Intelligence System

An end-to-end, production-oriented customer retention intelligence system. The
implemented foundation currently provides validated data ingestion, reproducible
development/hold-out splitting, exploratory notebooks, automated development-data
EDA, reproducible statistical reporting, and reusable feature preprocessing.
Classical and PyTorch ANN experiments with local MLflow tracking are
implemented. Development-only optimization, provisional calibration, and
threshold selection are available; recommendation capabilities remain planned.

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
- Five-fold stratified classical-model experiments with a dummy reference,
  Logistic Regression, Random Forest, XGBoost, class-weight comparisons,
  out-of-fold evaluation, and local MLflow tracking
- Nested-fold PyTorch ANN experiments with inner-validation early stopping,
  weighted-loss comparisons, out-of-fold evaluation, and MLflow tracking
- Development-only Optuna optimization, provisional model selection, sigmoid
  calibration comparison, and F1 threshold analysis
- Targeted XGBoost enhancement with fold-local feature and imbalance screens,
  extended search, and a precision-constrained recall operating point
- Development-only error, calibration, segment, learning-curve, and model-capacity
  diagnostics; frozen selection in `configs/selected_model.yaml`

## Planned Capabilities

- Final customer churn probability model and held-out evaluation
- Customer risk segmentation
- Further deep learning architecture comparison
- Customer-level SHAP explanations
- LLM-powered Next-Best-Action recommendations
- Model versioning and registry
- REST API and interactive analytics dashboard
- Data drift and model performance monitoring
- Automated testing and CI/CD

## Current Technology Stack

Python 3.11, pandas, NumPy, SciPy, scikit-learn, Pandera, PyArrow,
Matplotlib, seaborn, XGBoost, PyTorch, Optuna, imbalanced-learn, MLflow,
DVC, Pytest, and Ruff.

## Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis,training]"
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
For classical model architecture, commands, metrics, and initial development
results, see [model training](docs/model_training.md).
For ANN architecture, nested CV, and measured comparisons, see
[ANN training](docs/ann_training.md).
For reproducible Optuna search, provisional model selection, calibration, and
threshold analysis, see [model optimization](docs/model_optimization.md).
For the targeted XGBoost enhancement and revised provisional selection, see
[XGBoost enhancement](docs/xgboost_enhancement.md).
The current development-only choice is baseline-feature, sigmoid-calibrated
XGBoost with 535 depth-one trees and a full-precision threshold of
`0.3304499866713542`. Phase 6A++ diagnostics retained this choice. The frozen
configuration records all hyperparameters, hashes, score conventions, and
source MLflow runs. Generated diagnostics are in
`reports/generated/model_diagnostics/` when the development artifacts are present.
The held-out split has not been evaluated.
Depth 2–5 raised training AP but reduced validation AP; additional trees
plateaued. Exact predictor profiles with conflicting labels and probability
overlap limit discrimination with the recorded features. Mean fold AP is
0.66637, while pooled OOF AP is 0.66301. Repeated development-fold selection
can make apparent improvements and the chosen operating point optimistic.

Historical experiment entry points and the local run browser are:

```bash
python -m retention.models.training --config params.yaml
python -m retention.models.ann_training --config params.yaml
python -m retention.models.optimization --config params.yaml
python -m retention.models.xgboost_enhancement --config params.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Development Checks

```bash
scripts/test_suite.sh -q
.venv/bin/coverage report -m
ruff check .
ruff format --check .
python -m pip check
git diff --check
```

`scripts/test_suite.sh` runs the complete suite once with statement and branch
coverage. It sets OpenMP, BLAS, and related native thread pools to one thread
for the test process. An unconstrained run exceeded an 80-second bound;
faulthandler found it spending time in XGBoost training and then PyTorch loss
evaluation. With controlled threads, the same tests completed in one run.
This setting applies to tests only and does not change model training code.

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
  mappings are fitted separately inside each training fold.
- No globally preprocessed dataset, resampling, or held-out model evaluation
  has been implemented.

## Project Status

- [x] Phase 1: Project initialization
- [x] Phase 2: Data ingestion and validation
- [x] Phase 3A: Development-data EDA and automated reporting
- [x] Phase 3B: Statistical hypothesis testing and automated reporting
- [x] Phase 3C: Feature engineering and preprocessing readiness
- [x] Phase 3D: Class imbalance comparison
- [x] Phase 4: Initial classical machine learning and MLflow tracking
- [x] Phase 5: Initial PyTorch ANN and MLflow experiments
- [x] Phase 6A: Development-only optimization and provisional selection
- [x] Phase 6A+: Targeted XGBoost enhancement and revised selection
- [x] Phase 6A++: Development-only diagnostics and retained selection
- [x] Phase 6B: Final training integration and held-out evaluation
- [x] Phase 7: PostgreSQL persistence and local integration verification

## Disclaimer

This project is an educational implementation using historical customer data.
Observed churn associations and monthly-charge exposure are descriptive. They do
not establish causal effects, realized revenue loss, customer lifetime value, or
retention-treatment effectiveness.

## Final model candidate

Phase 6B's frozen XGBoost candidate, independent held-out metrics,
reproducibility commands and limitations are documented in
[docs/final_model.md](docs/final_model.md). The [model card](docs/final_model_card.md)
summarizes intended use and monitoring. The candidate is registered under
the MLflow `candidate` alias; it has not been promoted to production.

## PostgreSQL persistence

Phase 7 database architecture, local PostgreSQL initialization, migrations,
synthetic persistence checks, and isolated test setup are documented in
[docs/database.md](docs/database.md). Database integration requires a local
PostgreSQL server and uses environment-provided credentials.
