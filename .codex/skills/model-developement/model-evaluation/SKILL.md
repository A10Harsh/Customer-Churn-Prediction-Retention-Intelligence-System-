---
name: model-evaluation
description: Calibrate and evaluate churn classifiers when validation decisions and a one-time final test report are required.
---

## Purpose and use cases

Turn a trained candidate into an evidence-backed evaluation with calibrated probabilities, an approval-aware cost threshold, subgroup diagnostics, and a sealed final report. All calibration and threshold choices use validation; the test set is opened once only after the candidate is frozen.

- Compare discrimination and calibration for trained candidates.
- Choose a cost-based threshold on validation data.
- Audit false negatives and subgroup performance.
- Produce the one-time final test report for a frozen model.
- Do not tune model hyperparameters; use `model-optimization`.
- Do not register or promote the winner; use `model-management`.

## Inputs and outputs

| Name | Type | Required | Description |
|---|---|---:|---|
| Trained pipeline | joblib/MLflow URI | Yes | Candidate frozen before test access. |
| Validation split | Parquet/DataFrame | Yes | Calibration, threshold, and model-choice data. |
| Test split | Parquet/DataFrame | Yes | Read once after model, calibrator, and threshold are frozen. |
| Evaluation config | YAML/dict | Yes | Seed, missed-churn cost, offer cost, subgroup columns, calibration method. |
| Training run ID | string | Yes | Parent lineage and model family from `model-training` or `model-optimization`. |
| Threshold approval | record/boolean | Conditional | Required before changing the configured threshold. |

Outputs:

- `artifacts/evaluation/<run_id>/validation_metrics.json` and `test_metrics.json`.
- `artifacts/evaluation/<run_id>/calibration.csv`, `threshold_curve.csv`, and `subgroup_metrics.csv`.
- MLflow evaluation run linked to the training run; plots logged under `evaluation/`.

## Execution

1. Load the frozen pipeline and validation data. Drop the target and identifier and assert the pipeline was not fitted on validation or test rows.

```python
import joblib, pandas as pd

pipeline = joblib.load(model_path)
valid = pd.read_parquet("data/processed/validation.parquet")
X_valid, y_valid = valid.drop(columns=["Churn", "customerID"]), valid.Churn.astype(int)
assert {"Churn", "customerID"}.isdisjoint(X_valid.columns)
```

2. Fit a sigmoid or isotonic calibrator on validation predictions only. Choose the method from validation CV or a predeclared sample-size rule, never test performance.

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

raw_valid = pipeline.predict_proba(X_valid)[:, 1]
if calibration_method == "isotonic":
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_valid, y_valid)
    p_valid = calibrator.predict(raw_valid)
else:
    calibrator = LogisticRegression(random_state=42).fit(
        raw_valid.reshape(-1, 1), y_valid
    )
    p_valid = calibrator.predict_proba(raw_valid.reshape(-1, 1))[:, 1]
```

3. Calculate the configured cost for each validation threshold and select the minimum. If it differs from the current configured threshold, stop and request explicit approval before adopting it.

```python
import numpy as np

thresholds = np.linspace(0.01, 0.99, 99)
costs = [
    missed_churn_cost * ((p_valid < t) & (y_valid == 1)).sum()
    + offer_cost * ((p_valid >= t) & (y_valid == 0)).sum()
    for t in thresholds
]
proposed_threshold = float(thresholds[int(np.argmin(costs))])
if proposed_threshold != configured_threshold and not threshold_approved:
    raise RuntimeError(
        f"Approval required to change threshold to {proposed_threshold:.2f}"
    )
```

4. Compute the full metric suite, Brier score, confusion counts, and subgroup support on validation. Report subgroup metrics only where configured minimum support is met.

```python
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

pred = (p_valid >= approved_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_valid, pred, labels=[0, 1]).ravel()
validation_metrics = {
    "roc_auc": roc_auc_score(y_valid, p_valid),
    "pr_auc": average_precision_score(y_valid, p_valid),
    "precision": precision_score(y_valid, pred, zero_division=0),
    "recall": recall_score(y_valid, pred, zero_division=0),
    "brier": brier_score_loss(y_valid, p_valid),
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp,
}
```

5. Freeze model URI, calibrator, threshold, code revision, and report schema. Then read the test split once, apply without refitting, and write the final test report.

```python
test = pd.read_parquet("data/processed/test.parquet")
X_test, y_test = test.drop(columns=["Churn", "customerID"]), test.Churn.astype(int)
raw_test = pipeline.predict_proba(X_test)[:, 1]
p_test = (
    calibrator.predict(raw_test)
    if calibration_method == "isotonic"
    else calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
)
test_pred = (p_test >= approved_threshold).astype(int)
test_metrics = {
    "roc_auc": roc_auc_score(y_test, p_test),
    "pr_auc": average_precision_score(y_test, p_test),
    "precision": precision_score(y_test, test_pred, zero_division=0),
    "recall": recall_score(y_test, test_pred, zero_division=0),
    "brier": brier_score_loss(y_test, p_test),
}
```

6. Log exact fields. Include `model_family`, `framework`, `candidate_role`, `baseline_run_id`, `seed`, `data_version`, `training_run_id`, calibration/threshold fields, and `test_evaluation_count`; metrics `validation_*` and `test_*` for ROC-AUC, PR-AUC, precision, recall, Brier plus confusion counts; artifacts `calibrator.joblib`, all JSON/CSV reports, calibration plot, PR curve, and confusion matrix.

```python
import mlflow

with mlflow.start_run(run_name="final-evaluation"):
    mlflow.log_params(
        {
            "model_family": model_family,
            "framework": "pytorch"
            if model_family == "ann"
            else ("xgboost" if model_family == "xgboost" else "scikit-learn"),
            "candidate_role": candidate_role,
            "baseline_run_id": baseline_run_id,
            "seed": 42,
            "data_version": data_version,
            "training_run_id": training_run_id,
            "calibration_method": calibration_method,
            "decision_threshold": approved_threshold,
            "threshold_rule": "minimum_configured_cost",
            "missed_churn_cost": missed_churn_cost,
            "offer_cost": offer_cost,
            "test_evaluation_count": 1,
        }
    )
    mlflow.log_metrics(
        {f"validation_{k}": float(v) for k, v in validation_metrics.items()}
    )
    mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
    mlflow.log_artifacts(report_dir, artifact_path="evaluation")
```

7. Hand off the sealed report to `model-interpretability` for explanation checks and to `model-management` for comparison and approval-gated registration.

## Validation checks and best practices

- [ ] Assert calibration fit received validation rows only and no test labels.
- [ ] Assert threshold cost uses validation only and matches the configured cost formula exactly.
- [ ] Verify explicit approval exists if the adopted threshold changed.
- [ ] Verify test evaluation count equals `1` for the candidate final model.
- [ ] Verify ROC-AUC, PR-AUC, precision, recall, and Brier are finite for validation and test.
- [ ] Verify subgroup rows meet minimum support and include prevalence, support, false-negative rate, and required metrics.
- [ ] Verify all report files exist and the MLflow run links to the immutable training run.
- Pitfall: calibrating an already fitted model with test; fix by fit the calibrator on validation predictions only.
- Pitfall: selecting a threshold from test; fix by seal the validation threshold before test access.
- Pitfall: hiding zero-support failures; fix by log support and use explicit `zero_division=0`.
- Hard rules: split first, fit preprocessing on training only, exclude `Churn` and `customerID`, and fix seeds.
- Hard rules: use test once for final reporting; never tune from it or report accuracy alone.
- Hard rules: never delete or overwrite data, artifacts, model versions, or migrations.
- Hard rules: stop for approval before threshold change, promotion, deployment, remote push, or cost; never read `.env` or expose PII/customer IDs to an LLM.
