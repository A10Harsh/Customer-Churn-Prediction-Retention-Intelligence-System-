---
name: model-selection
description: Compare leakage-safe churn classifiers and select a justified shortlist when a baseline or candidate model family must be chosen.
---

## Purpose and use cases

Choose a defensible model family before expensive tuning. Establish a simple baseline first, compare candidates with identical stratified folds, quantify uncertainty, and record why the winner fits the operating constraints.

- Establish the required logistic-regression baseline.
- Compare Random Forest, XGBoost, and direct PyTorch ANN challengers on identical folds.
- Revisit the shortlist after material feature or data changes.
- Resolve a tradeoff between PR-AUC, calibration, latency, and interpretability.
- Do not use for hyperparameter search; use `model-optimization`.
- Do not use for the one-time final test report; use `model-evaluation`.

## Inputs and outputs

| Name | Type | Required | Description |
|---|---|---:|---|
| Training split | Parquet/CSV or DataFrame | Yes | Training rows created before fitting, with `Churn` and `customerID` retained only for separation. |
| Validation split | Parquet/CSV or DataFrame | Yes | Untouched selection holdout; never merged with test data. |
| Preprocessing pipeline | sklearn transformer | Yes | Train-fitted imputation/encoding/scaling definition from data preparation. |
| Selection config | YAML/dict | Yes | Seed, stratified fold count, candidates, primary metric, and runtime limits. |
| Test split | Parquet/CSV | Yes | Presence is verified, but contents are not used during selection. |
| Data version | DVC hash/string | Yes | Immutable input revision for lineage. |

Outputs:

- `artifacts/model_selection/cv_results.csv`: fold-level ROC-AUC, PR-AUC, precision, and recall.
- `artifacts/model_selection/selection.json`: shortlist, confidence intervals, constraints, and rationale.
- MLflow runs in experiment `churn-retention`, one parent and one child per candidate.

## Execution

1. Load the predefined splits and separate protected columns before any fit. Assert the test split exists without reading it into candidate comparison logic.

```python
import pandas as pd

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/validation.parquet")
X_train = train.drop(columns=["Churn", "customerID"])
y_train = train["Churn"].astype(int)
X_valid = valid.drop(columns=["Churn", "customerID"])
y_valid = valid["Churn"].astype(int)
assert {"Churn", "customerID"}.isdisjoint(X_train.columns)
assert __import__("pathlib").Path("data/processed/test.parquet").exists()
```

2. Fit the untuned logistic-regression baseline first. Then evaluate untuned Random Forest, XGBoost, and direct PyTorch ANN challengers with the same Pipeline, splits, and stratified folds. The ANN adapter must wrap `torch.nn.Module`, implement sklearn cloning plus `predict_proba`, and accept dense `float32` transformed input.

```python
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from src.models.torch_ann import TorchANNClassifier
from xgboost import XGBClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
# fmt: off
candidates = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=42),
    "xgboost": XGBClassifier(n_estimators=300, scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42),
    "ann": TorchANNClassifier(hidden_dims=(64, 32), max_epochs=100, patience=10,
                              batch_size=64, random_state=42, device="cpu"),
}
scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "precision": "precision", "recall": "recall"}
scores_by_candidate = {}
for name, estimator in candidates.items():
    pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
    scores_by_candidate[name] = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
# fmt: on
```

3. Compute a mean and 95% t-interval for every metric from fold scores. Rank by mean CV PR-AUC; use configured latency, calibration, and interpretability limits only as tie-breakers.

```python
import numpy as np
from scipy.stats import t

ci95_by_candidate = {}
for name, candidate_scores in scores_by_candidate.items():
    for metric in scoring:
        values = candidate_scores[f"test_{metric}"]
        half = float(
            t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        )
        ci95_by_candidate[(name, metric)] = [
            float(values.mean() - half),
            float(values.mean() + half),
        ]
```

4. Fit each candidate on training only and report validation metrics at threshold `0.5`; do not choose a new threshold here. Keep logistic regression labeled as the baseline regardless of which challenger ranks highest. Delegate calibration and thresholding to `model-evaluation`.

```python
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

validation_by_candidate = {}
# fmt: off
for name, estimator in candidates.items():
    pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)]).fit(X_train, y_train)
    p = pipe.predict_proba(X_valid)[:, 1]
    pred = (p >= 0.5).astype(int)
    validation_by_candidate[name] = {
        "pr_auc": average_precision_score(y_valid, p), "roc_auc": roc_auc_score(y_valid, p),
        "precision": precision_score(y_valid, pred, zero_division=0),
        "recall": recall_score(y_valid, pred, zero_division=0),
    }
# fmt: on
```

5. Log exact lineage and comparison fields. Save tables before logging them; never overwrite an existing run directory.

```python
import mlflow

# fmt: off
with mlflow.start_run(run_name="model-selection"):
    for name in ("logistic_regression", "random_forest", "xgboost", "ann"):
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params({"seed": 42, "data_version": data_version, "candidate": name,
                "candidate_role": "baseline" if name == "logistic_regression" else "challenger",
                "cv_strategy": "StratifiedKFold", "cv_folds": 5,
                "selection_metric": "pr_auc", "decision_threshold": 0.5})
            mlflow.log_metrics({f"cv_mean_{k}": float(scores_by_candidate[name][f"test_{k}"].mean())
                                for k in scoring})
            mlflow.log_metrics({f"cv_{k}_ci95_{bound}": ci95_by_candidate[(name, k)][i]
                                for k in scoring for i, bound in enumerate(("low", "high"))})
            mlflow.log_metrics({f"validation_{k}": float(v)
                                for k, v in validation_by_candidate[name].items()})
            mlflow.log_artifacts("artifacts/model_selection", artifact_path="model_selection")
# fmt: on
```

6. Write `selection.json` with the baseline, shortlist, rejected candidates, CI overlap, runtime, constraints, and selection rationale. Hand off the selected family to `model-training`, then `model-optimization`; send final candidates to `model-evaluation`.

## Validation checks and best practices

- [ ] Assert `Churn` and `customerID` are absent from every candidate's `X` columns.
- [ ] Assert all candidates use identical fold indices and seed `42` (or the configured fixed seed).
- [ ] Verify the untuned logistic-regression baseline run predates all challenger and optimization runs.
- [ ] Verify Random Forest, XGBoost, and PyTorch ANN use the baseline's exact split and fold indices.
- [ ] Verify ANN estimator module is a `torch.nn.Module` and no `MLPClassifier` is imported.
- [ ] Verify every candidate has 5 fold rows and finite PR-AUC, ROC-AUC, precision, and recall.
- [ ] Verify `cv_results.csv` and `selection.json` exist and are logged to MLflow.
- [ ] Verify the selected candidate has the highest eligible mean CV PR-AUC or a documented tie-break.
- Pitfall: comparing different folds; fix by materializing and reusing the same splitter indices.
- Pitfall: treating overlapping CIs as proof of superiority; fix by documenting uncertainty and operational tie-breaks.
- Pitfall: searching many parameters during selection; fix by moving that work to `model-optimization`.
- Hard rules: split before fitting; keep preprocessing inside the training-only Pipeline; use stratified splits and fixed seeds.
- Hard rules: never select with test results, never report accuracy alone, and never fabricate metrics.
- Hard rules: never delete or overwrite data, DVC/MLflow artifacts, registered models, or migrations.
- Hard rules: stop for approval before threshold changes, promotion, deployment, remote push, or cost; never read `.env` values or expose PII/customer IDs to an LLM.
