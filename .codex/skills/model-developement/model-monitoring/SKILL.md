---
name: model-monitoring
description: Monitor deployed churn models for drift and performance decay when production batches need alerts, retraining evidence, or incident analysis.
---

## Purpose and use cases

Compare production batches with an approved reference window using PSI, KS, prediction drift, and delayed-label performance. Apply predeclared alert thresholds, distinguish data-quality incidents from model decay, and recommend retraining without triggering it automatically.

- Generate scheduled input and prediction drift reports.
- Detect ROC-AUC, PR-AUC, precision, or recall decay after labels arrive.
- Evaluate alert thresholds and retraining triggers.
- Investigate a sudden segment or schema shift.
- Do not modify the serving release; use `model-deployment` with approval.
- Do not retrain or promote automatically; hand off to `model-selection` and `model-management`.

## Inputs and outputs

| Name | Type | Required | Description |
|---|---|---:|---|
| Reference data | Parquet/DataFrame | Yes | Approved training/validation reference with input schema and predictions. |
| Current batch | Parquet/DataFrame | Yes | Identifier-separated production inputs, predictions, timestamps, and optional labels. |
| Monitoring config | YAML/dict | Yes | Windows, bins, PSI/KS limits, metric-decay limits, minimum sample sizes, trigger policy. |
| Model metadata | JSON/MLflow URI | Yes | Version, run ID, data version, schema, threshold, deployment ID. |
| Delayed labels | Parquet/DataFrame | Conditional | Outcomes joined outside the report by an approved internal key; identifiers are never logged. |

Outputs:

- `monitoring/reports/<timestamp>/drift.json` and `performance.json`.
- `monitoring/reports/<timestamp>/feature_drift.csv` and `alerts.json`.
- Evidently HTML/JSON report when configured, plus an MLflow monitoring run.
- Retraining recommendation with evidence; no automatic training, promotion, or deployment.

## Execution

1. Validate schema, model version, time window, sample size, and identifier exclusion before statistics. Quarantine incompatible batches without deleting them.

```python
import pandas as pd

protected = {"Churn", "customerID"}
reference = pd.read_parquet(reference_path)
current = pd.read_parquet(current_path)
for frame in (reference, current):
    assert protected.isdisjoint(frame.columns)
assert set(input_columns) <= set(reference) & set(current)
assert len(current) >= min_batch_rows
```

2. Compute PSI with reference-derived bins and smoothing. Flag population drift only from configured thresholds and minimum support.

```python
import numpy as np


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    r = pd.cut(ref, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    c = pd.cut(cur, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    r, c = r.align(c, fill_value=0)
    r, c = r.clip(1e-6), c.clip(1e-6)
    return float(((c - r) * np.log(c / r)).sum())


psi_value = psi(reference[column], current[column])
```

3. Compute two-sample KS for continuous inputs, apply configured multiplicity correction, and report effect size with p-value. Use categorical proportion distance for categorical fields.

```python
from scipy.stats import ks_2samp

ks = ks_2samp(reference[column].dropna(), current[column].dropna(), method="auto")
ks_drift = ks.statistic >= ks_stat_threshold and ks.pvalue <= corrected_alpha
```

4. When labels have matured, compute the same frozen-threshold metric suite and compare with approved reference metrics. Do not retune the threshold from production data in this skill.

```python
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

p, y = labeled["churn_probability"], labeled["Churn"].astype(int)
pred = (p >= deployed_threshold).astype(int)
current_metrics = {
    "roc_auc": roc_auc_score(y, p),
    "pr_auc": average_precision_score(y, p),
    "precision": precision_score(y, pred, zero_division=0),
    "recall": recall_score(y, pred, zero_division=0),
}
decay = {k: reference_metrics[k] - v for k, v in current_metrics.items()}
```

5. Evaluate predeclared alert levels. A retraining recommendation requires sufficient samples plus either sustained drift for the configured consecutive windows or metric decay beyond a configured limit; it never launches training automatically.

```python
retrain_recommended = len(current) >= retrain_min_rows and (
    consecutive_drift_windows >= required_windows
    or decay.get("pr_auc", 0.0) >= max_pr_auc_decay
    or decay.get("recall", 0.0) >= max_recall_decay
)
```

6. Optionally create the project-standard Evidently report using the same reference/current frames and save both HTML and machine-readable JSON. Keep the custom PSI/KS table as the alert contract.

7. Log exact MLflow fields: params `model_name`, `model_family`, `model_version`, `model_run_id`, deployment/reference/window/threshold settings; metrics batch size, drift counts and maxima, prediction PSI, current and decay ROC-AUC/PR-AUC/precision/recall when labels exist, alerts, and retraining recommendation; artifacts drift/performance JSON, drift table, alerts, schema result, and Evidently reports.

```python
import mlflow

with mlflow.start_run(run_name="production-monitoring"):
    mlflow.log_params(
        {
            "model_name": "churn-classifier",
            "model_family": model_family,
            "model_version": model_version,
            "model_run_id": model_run_id,
            "deployment_id": deployment_id,
            "reference_data_version": reference_data_version,
            "window_start": window_start,
            "window_end": window_end,
            "deployed_threshold": deployed_threshold,
            "psi_threshold": psi_threshold,
            "ks_stat_threshold": ks_stat_threshold,
            "corrected_alpha": corrected_alpha,
            "min_batch_rows": min_batch_rows,
            "required_windows": required_windows,
        }
    )
    mlflow.log_metrics(
        {
            "batch_rows": len(current),
            "drifted_field_count": drifted_count,
            "max_field_psi": max_psi,
            "max_field_ks": max_ks,
            "alert_count": len(alerts),
            "retrain_recommended": int(retrain_recommended),
        }
    )
    mlflow.log_artifacts(report_dir, artifact_path="monitoring")
```

8. Handoff: receive deployment metadata from `model-deployment`; route schema/data failures to data preparation, retraining evidence to `model-selection`, and approved model changes through `model-management` and `model-deployment`.

## Validation checks and best practices

- [ ] Assert current/reference schemas match and exclude `Churn`, `customerID`, and PII from statistics/logs.
- [ ] Verify PSI bins come only from reference data and every PSI/KS value is finite.
- [ ] Verify KS alerts use corrected alpha and configured effect-size threshold.
- [ ] Verify performance metrics are computed only after label maturity and minimum sample count.
- [ ] Verify real ROC-AUC, PR-AUC, precision, and recall use the deployed threshold and immutable model version.
- [ ] Verify every alert names window, field/metric, observed value, threshold, severity, and model version.
- [ ] Verify reports exist under a new timestamped directory and no prior report was overwritten.
- Pitfall: alerting on p-value alone; fix by require an effect-size threshold and adequate support.
- Pitfall: reacting to one noisy window; fix by require configured consecutive windows.
- Pitfall: auto-retraining on drift; fix by emit evidence and wait for approval-governed lifecycle steps.
- Hard rules: upstream training must split first, fit preprocessing on train only, exclude target/ID, stratify, and fix seeds.
- Hard rules: never tune thresholds from monitoring/test data or report accuracy alone.
- Hard rules: never delete or overwrite data, DVC/MLflow artifacts, models, reports, or migrations.
- Hard rules: stop for approval before retraining action, threshold change, promotion, deployment, push, or cost; never read `.env` or send PII/customer IDs to an LLM.
