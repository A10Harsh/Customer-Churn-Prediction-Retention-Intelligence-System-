---
name: model-training
description: Train reproducible churn-classification pipelines when a selected model family must be fitted, logged, and saved without leakage.
---

## Purpose and use cases

Fit the required logistic-regression baseline or one selected challenger with preprocessing encapsulated in one serializable pipeline. Handle model-specific imbalance and early-stopping policies, reproducible logging, and immutable artifacts while leaving tuning and final test decisions to their dedicated skills.

- Train the logistic-regression baseline before any challenger.
- Train Random Forest, XGBoost, or direct PyTorch ANN experiments against that baseline.
- Refit an approved parameter set on the training partition.
- Compare class weighting with an unweighted training run.
- Reproduce a run from its data revision, config, and seed.
- Do not search hyperparameters; use `model-optimization`.
- Do not calibrate, set thresholds, or consume the test set; use `model-evaluation`.

## Inputs and outputs

| Name | Type | Required | Description |
|---|---|---:|---|
| Training split | Parquet/DataFrame | Yes | The only rows fitted by preprocessing and estimator. |
| Validation split | Parquet/DataFrame | Yes | Used for provisional metrics only; never fitted. |
| Pipeline definition | sklearn Pipeline | Yes | Imputer, encoder, scaler if needed, and estimator. |
| Training config | YAML/dict | Yes | Model family, seed, approved parameters, imbalance method, early-stopping policy, paths. |
| Selection record | JSON/MLflow run ID | Yes | Candidate family chosen by `model-selection`. |
| Data version | DVC hash/string | Yes | Immutable split/input revision. |

Outputs:

- `artifacts/training/<run_id>/model.joblib`: fitted sklearn-compatible pipeline; ANN runs also save `torch_state_dict.pt`.
- `artifacts/training/<run_id>/training_summary.json`: parameters, row counts, class prevalence, and metrics.
- MLflow training run with model signature and input example.

## Execution

1. Load training and validation splits, remove `Churn` and `customerID`, and verify stratification and disjoint row identities before fitting.

```python
import pandas as pd

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/validation.parquet")
assert set(train.customerID).isdisjoint(valid.customerID)
X_train, y_train = train.drop(columns=["Churn", "customerID"]), train.Churn.astype(int)
X_valid, y_valid = valid.drop(columns=["Churn", "customerID"]), valid.Churn.astype(int)
assert {"Churn", "customerID"}.isdisjoint(X_train.columns)
```

2. Derive imbalance values from training labels only. Use balanced class weights for logistic regression and Random Forest, `scale_pos_weight` for XGBoost, and PyTorch `BCEWithLogitsLoss(pos_weight=...)` inside the ANN adapter. Keep resampling and early stopping inside fold-local training; never resample validation or test rows.

```python
neg, pos = y_train.value_counts().reindex([0, 1], fill_value=0)
scale_pos_weight = float(neg / pos)
assert pos > 0 and neg > 0
```

3. Build the selected estimator and fit the complete Pipeline on training only. Before ANN training, add `torch` to `pyproject.toml` and implement `src/models/torch_ann.py`: a `BaseEstimator`/`ClassifierMixin` adapter around `torch.nn.Module` with deterministic seeds, internal stratified early stopping, `BCEWithLogitsLoss`, and two-column `predict_proba`. For XGBoost, use the CV-selected `n_estimators`; do not bypass the Pipeline.

```python
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.models.torch_ann import TorchANNClassifier
from xgboost import XGBClassifier

# fmt: off
builders = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "random_forest": lambda: RandomForestClassifier(class_weight="balanced_subsample", random_state=42, **model_params),
    "xgboost": lambda: XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42, **model_params),
    "ann": lambda: TorchANNClassifier(pos_weight=scale_pos_weight, random_state=42,
                                      validation_fraction=0.15, device="cpu", **model_params),
}
if model_family != "logistic_regression":
    assert baseline_run_id, "Logistic-regression baseline must exist first"
pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", builders[model_family]())])
# fmt: on
pipeline.fit(X_train, y_train)
```

4. Compute provisional validation metrics at the unchanged configured threshold. Do not tune that threshold in this skill.

```python
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

p = pipeline.predict_proba(X_valid)[:, 1]
pred = (p >= configured_threshold).astype(int)
metrics = {
    "validation_pr_auc": average_precision_score(y_valid, p),
    "validation_roc_auc": roc_auc_score(y_valid, p),
    "validation_precision": precision_score(y_valid, pred, zero_division=0),
    "validation_recall": recall_score(y_valid, pred, zero_division=0),
}
```

5. Save to a new run-specific directory and log exact fields and artifacts to MLflow.

```python
import joblib, mlflow, torch
from mlflow.models import infer_signature

with mlflow.start_run(run_name=f"{model_family}-training") as run:
    out = f"artifacts/training/{run.info.run_id}"
    __import__("pathlib").Path(out).mkdir(parents=True, exist_ok=False)
    joblib.dump(pipeline, f"{out}/model.joblib")
    if model_family == "ann":
        torch.save(
            pipeline.named_steps["model"].module_.state_dict(),
            f"{out}/torch_state_dict.pt",
        )
    # fmt: off
    mlflow.log_params(
        {
            "seed": 42,
            "data_version": data_version,
            "model_family": model_family,
            "framework": "pytorch" if model_family == "ann" else ("xgboost" if model_family == "xgboost" else "scikit-learn"),
            "candidate_role": "baseline" if model_family == "logistic_regression" else "challenger",
            "baseline_run_id": baseline_run_id or "self",
            "imbalance_method": imbalance_method,
            "scale_pos_weight": scale_pos_weight,
            "early_stopping_policy": "internal_training_split" if model_family == "ann" else "none",
            "decision_threshold": configured_threshold,
            **model_params,
        }
    )
    # fmt: on
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        pipeline,
        "model",
        signature=infer_signature(X_valid, p),
        input_example=X_valid.head(3),
    )
    mlflow.log_artifacts(out, artifact_path="training")
```

6. Record the exact DVC revision without mutating tracked data. Hand the logistic-regression baseline and challenger runs to `model-optimization`; send trained finalists to `model-evaluation` and approved artifacts to `model-management`.

## Validation checks and best practices

- [ ] Assert training and validation customer IDs are disjoint before dropping identifiers.
- [ ] Assert `Churn` and `customerID` are absent from model inputs and saved signature.
- [ ] Verify the preprocessor's fitted row count or fit audit equals the training row count only.
- [ ] Verify the MLflow run logs seed, data version, full estimator parameters, imbalance method, and threshold.
- [ ] Verify logistic regression is tagged `baseline` and predates Random Forest, XGBoost, and PyTorch ANN runs.
- [ ] Verify ANN artifacts include `torch_state_dict.pt`, framework `pytorch`, and a CPU-reload probability match within `1e-6`.
- [ ] Verify validation ROC-AUC, PR-AUC, precision, and recall are finite real-run values.
- [ ] Verify `model.joblib` reloads and matches five validation probabilities within `1e-6` for ANN or `1e-12` otherwise.
- [ ] Verify the output directory was created with `exist_ok=False` and no artifact was overwritten.
- Pitfall: fitting preprocessing before the split; fix by cloning and fitting it only on `X_train`.
- Pitfall: early stopping on test; fix by using validation only and reserving test for `model-evaluation`.
- Pitfall: combining class weights and resampling blindly; fix by compare one configured method at a time in CV.
- Hard rules: use fixed seeds, stratified splits, training-only preprocessing, and no target/identifier features.
- Hard rules: never select parameters or thresholds using test; never report accuracy without the required metric suite.
- Hard rules: never delete or overwrite data, artifacts, registry versions, or migrations.
- Hard rules: stop for human approval before threshold changes, promotion, deployment, push, or cost; never read `.env` or send PII/customer IDs to an LLM.
