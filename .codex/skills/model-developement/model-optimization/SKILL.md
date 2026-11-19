---
name: model-optimization
description: Tune churn classifiers with Optuna and leakage-safe nested validation when an established baseline needs bounded hyperparameter optimization.
---

## Purpose and use cases

Tune each challenger within a declared trial and compute budget after the logistic-regression baseline is logged. Search only training folds, compare Random Forest, XGBoost, and direct PyTorch ANN separately against the same baseline, and preserve every trial.

- Tune Random Forest, XGBoost, and PyTorch ANN in separate studies.
- Define bounded, domain-aware search spaces.
- Run nested CV when unbiased tuning estimates are required.
- Do not choose families or thresholds here; use `model-selection` before and `model-evaluation` after.

## Inputs and outputs

| Name | Type | Required | Description |
|---|---|---:|---|
| Training split | Parquet/DataFrame | Yes | Source for inner/outer stratified folds; no preprocessing is pre-fitted. |
| Validation split | Parquet/DataFrame | Yes | Single comparison against the frozen baseline after tuning. |
| Pipeline definition | sklearn Pipeline | Yes | All preprocessing plus candidate estimator. |
| Baseline run | MLflow run ID | Yes | Logged comparison anchor from `model-training`. |
| Optimization config | YAML/dict | Yes | Seed, study name, search space, CV folds, direction, timeout, trial budget. |
| Optuna storage | SQLite/PostgreSQL URI | Yes | Persistent study storage; credentials supplied by runtime, never read from `.env`. |
| Data version | DVC hash/string | Yes | Immutable input revision. |

Outputs:
- Persistent studies `churn-<family>-<data_version>`, files under `artifacts/optimization/<study>/`, and MLflow parent/trial runs with the best Pipeline.

## Execution

1. Verify the logistic-regression baseline MLflow run completed first and contains validation PR-AUC, ROC-AUC, precision, and recall. Load only train and validation splits; assert the test path exists but do not read it.

```python
from pathlib import Path
import mlflow, pandas as pd

baseline = mlflow.get_run(baseline_run_id)
assert baseline.data.params["model_family"] == "logistic_regression"
assert baseline.data.metrics["validation_pr_auc"] >= 0
train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/validation.parquet")
assert Path("data/processed/test.parquet").exists()
X, y = train.drop(columns=["Churn", "customerID"]), train.Churn.astype(int)
```

2. Run one study per challenger. Declare bounded family-specific spaces and rebuild the complete Pipeline for every trial so all preprocessing fits inside each training fold.

```python
import optuna
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from src.models.torch_ann import TorchANNClassifier
from xgboost import XGBClassifier

inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
assert model_family in {"random_forest", "xgboost", "ann"}


# fmt: off
def suggest(trial, family):
    if family == "random_forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 1200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20)}
    if family == "xgboost":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0)}
    return {"hidden_dim": trial.suggest_int("hidden_dim", 16, 128),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)}

def make_pipeline(family, params, y_fit):
    if family == "random_forest":
        model = RandomForestClassifier(class_weight="balanced_subsample", random_state=42, **params)
    elif family == "xgboost":
        weight = float((y_fit == 0).sum() / (y_fit == 1).sum())
        model = XGBClassifier(scale_pos_weight=weight, eval_metric="aucpr", random_state=42, **params)
    else:
        weight = float((y_fit == 0).sum() / (y_fit == 1).sum())
        model = TorchANNClassifier(hidden_dims=(params.pop("hidden_dim"),), pos_weight=weight,
                                   max_epochs=100, patience=10, random_state=42, device="cpu", **params)
    return Pipeline([("preprocess", clone(preprocessor)), ("model", model)])

def objective(trial, X_data=X, y_data=y):
    candidate = make_pipeline(model_family, suggest(trial, model_family), y_data)
    scores = cross_validate(
        candidate, X_data, y_data, cv=inner, scoring="average_precision", n_jobs=1
    )
    trial.set_user_attr("fold_pr_auc", scores["test_score"].tolist())
    return float(scores["test_score"].mean())
# fmt: on
```

3. When nested CV is configured, create a fresh inner study on each outer-training fold, refit its winner on that fold, and score only the outer holdout. Otherwise retain validation for the sole post-search comparison. Never use test as an outer fold.

```python
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
outer_scores = []
# fmt: off
for fold, (train_idx, holdout_idx) in enumerate(outer.split(X, y)):
    fold_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42 + fold))
    fold_study.optimize(lambda t: objective(t, X.iloc[train_idx], y.iloc[train_idx]),
                        n_trials=nested_trial_budget_per_fold)
    fold_model = make_pipeline(model_family, fold_study.best_params.copy(), y.iloc[train_idx])
    fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    outer_scores.append(average_precision_score(
        y.iloc[holdout_idx], fold_model.predict_proba(X.iloc[holdout_idx])[:, 1]))
# fmt: on
```

4. Create or resume the persistent study and enforce both trial and time budgets. Set the sampler seed and pruning policy in code/config.

```python
study_name = f"churn-{model_family}-{data_version}"
study = optuna.create_study(
    study_name=study_name,
    storage=storage_uri,
    direction="maximize",
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
)
remaining = max(0, trial_budget - len(study.trials))
study.optimize(objective, n_trials=remaining, timeout=timeout_seconds)
assert len(study.trials) <= trial_budget
```

5. Fit the best Pipeline on training only, compare once on validation at the unchanged threshold, and require improvement over baseline by the configured minimum PR-AUC delta or retain the baseline.

```python
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

best = make_pipeline(model_family, study.best_params.copy(), y)
best.fit(X, y)
Xv, yv = valid.drop(columns=["Churn", "customerID"]), valid.Churn.astype(int)
p = best.predict_proba(Xv)[:, 1]
pred = (p >= configured_threshold).astype(int)
metrics = {
    "validation_pr_auc": average_precision_score(yv, p),
    "validation_roc_auc": roc_auc_score(yv, p),
    "validation_precision": precision_score(yv, pred, zero_division=0),
    "validation_recall": recall_score(yv, pred, zero_division=0),
}
accepted = (
    metrics["validation_pr_auc"]
    >= baseline.data.metrics["validation_pr_auc"] + min_pr_auc_delta
)
```

6. Log exact fields: params `model_family`, `seed`, `data_version`, `baseline_run_id`, `study_name`, `sampler`, `pruner`, budgets, folds, `selection_metric`, `decision_threshold`, and every best parameter; metrics `completed_trials`, `pruned_trials`, `best_cv_pr_auc`, nested-CV mean/std PR-AUC, four `validation_*` metrics, and `validation_pr_auc_delta`; artifacts trials, parameters, search space, comparison, and accepted pipeline.

```python
# fmt: off
with mlflow.start_run(run_name=f"{model_family}-optuna"):
    mlflow.log_params({"seed": 42, "model_family": model_family, "data_version": data_version,
        "baseline_run_id": baseline_run_id, "study_name": study_name, "sampler": "TPESampler",
        "pruner": "MedianPruner", "trial_budget": trial_budget,
        "nested_trial_budget_per_fold": nested_trial_budget_per_fold, "timeout_seconds": timeout_seconds,
        "inner_folds": 5, "outer_folds": 5, "selection_metric": "pr_auc",
        "decision_threshold": configured_threshold,
        **{f"best_{k}": v for k, v in study.best_params.items()}})
    mlflow.log_metrics({**metrics, "completed_trials": len(study.get_trials()),
        "pruned_trials": sum(t.state.name == "PRUNED" for t in study.trials),
        "best_cv_pr_auc": study.best_value, "nested_cv_mean_pr_auc": float(np.mean(outer_scores)),
        "nested_cv_std_pr_auc": float(np.std(outer_scores, ddof=1)),
        "validation_pr_auc_delta": metrics["validation_pr_auc"] - baseline.data.metrics["validation_pr_auc"]})
    mlflow.log_artifacts(artifact_dir, artifact_path="optimization")
# fmt: on
```

7. Hand off from `model-training`; return an accepted candidate to `model-evaluation`, then `model-management`.

## Validation checks and best practices

- [ ] Verify the baseline predates separate Random Forest, XGBoost, and PyTorch ANN studies with identical folds/budgets; ANN trials must instantiate `torch.nn.Module` and log device/seed.
- [ ] Assert every Pipeline clone refits preprocessing and excludes `Churn` and `customerID` in every fold.
- [ ] Verify trial count respects the budget/timeout and each completed trial stores five fold PR-AUC values plus its seed.
- [ ] Verify comparison includes real ROC-AUC, PR-AUC, precision, and recall for baseline and candidate.
- [ ] Verify no test file read occurs in the optimization code path.
- Pitfalls: tune the full Pipeline inside CV; require the configured PR-AUC delta or preserve the baseline.
- Hard rules: stratify, fix seeds, split before fit, and fit every transform on fold-training rows only.
- Hard rules: test is final-report-only; never select from it or report accuracy alone.
- Hard rules: never delete or overwrite data, DVC/MLflow artifacts, model versions, or migrations.
- Hard rules: stop for approval before threshold changes, promotion, deployment, push, or cost; never read `.env` or expose PII/customer IDs to an LLM.
