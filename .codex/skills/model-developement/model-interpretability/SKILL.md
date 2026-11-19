---
name: model-interpretability
description: Explain churn models globally and per prediction when feature effects, explanation stability, and suspicious leakage signals must be audited.
---

## Purpose and use cases

Produce global and local explanations for a frozen tabular churn pipeline without exposing customer identifiers. Use permutation importance as a model-agnostic check, SHAP where supported, and stability analysis to distinguish robust drivers from leakage or sampling artifacts.

- Create global driver rankings for a trained churn model.
- Generate top local drivers for a prediction response.
- Compare permutation importance with SHAP rankings.
- Investigate implausibly predictive or unstable fields for leakage.
- Do not select fields before fitting; use `select-features` and `detect-data-leakage`.
- Do not evaluate final discrimination or thresholds; use `model-evaluation`.

## Inputs and outputs

| Name | Type | Required | Description |
|---|---|---:|---|
| Frozen pipeline | joblib/MLflow URI | Yes | Evaluated preprocessing plus estimator. |
| Validation sample | Parquet/DataFrame | Yes | Stratified, bounded explanation sample; never test-driven selection. |
| Lineage map | JSON/list | Yes | Mapping from transformed columns to approved raw fields. |
| Interpretability config | YAML/dict | Yes | Seed, sample size, repeats, top-k, stability threshold, allowed fields. |
| Evaluation run ID | string | Yes | Immutable metrics and threshold lineage. |
| Background sample | array/DataFrame | Conditional | Training-only background for model-specific SHAP explainers. |

Outputs:

- `artifacts/interpretability/<run_id>/permutation_importance.csv`.
- `artifacts/interpretability/<run_id>/shap_global.csv` and `local_explanations.jsonl` without identifiers.
- `artifacts/interpretability/<run_id>/stability.csv` and `leakage_review.json`.
- MLflow interpretability run with summary plots and source run IDs.

## Execution

1. Load the frozen Pipeline and a configured validation sample. Separate protected columns before sampling and assert they are absent from explanation inputs and lineage.

```python
import joblib, pandas as pd

protected = {"Churn", "customerID"}
pipeline = joblib.load(model_path)
valid = pd.read_parquet("data/processed/validation.parquet")
X, y = valid.drop(columns=list(protected)), valid["Churn"].astype(int)
sample = X.sample(n=min(sample_size, len(X)), random_state=42)
assert protected.isdisjoint(sample.columns)
assert all("customer" not in name.lower() for name in lineage_map)
```

2. Compute model-agnostic permutation importance on validation with PR-AUC scoring and fixed repeats. Keep the complete Pipeline intact so raw-column permutations reflect serving behavior.

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(
    pipeline,
    sample,
    y.loc[sample.index],
    scoring="average_precision",
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)
permutation_df = pd.DataFrame(
    {
        "column": sample.columns,
        "mean_pr_auc_drop": perm.importances_mean,
        "std_pr_auc_drop": perm.importances_std,
    }
).sort_values("mean_pr_auc_drop", ascending=False)
```

3. Compute SHAP values on transformed, identifier-free columns. Use `LinearExplainer` for logistic regression, `TreeExplainer` for Random Forest/XGBoost, and bounded `PermutationExplainer` calls against the PyTorch adapter's `predict_proba`. Build the background from training rows only.

```python
import numpy as np, shap

pre, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
Xt = pre.transform(sample)
column_names = pre.get_feature_names_out()
background = pre.transform(training_background)
if model_family == "logistic_regression":
    explainer_type, explainer = (
        "LinearExplainer",
        shap.LinearExplainer(model, background),
    )
elif model_family in {"random_forest", "xgboost"}:
    explainer_type = "TreeExplainer"
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional"
    )
else:
    explainer_type = "PermutationExplainer"
    explainer = shap.PermutationExplainer(
        lambda z: model.predict_proba(z)[:, 1], background
    )
values = (
    explainer(Xt, check_additivity=False)
    if explainer_type == "TreeExplainer"
    else explainer(Xt)
)
shap_global = pd.DataFrame(
    {"column": column_names, "mean_abs_shap": np.abs(values.values).mean(axis=0)}
).sort_values("mean_abs_shap", ascending=False)
```

4. Produce local top-k explanations with column name, direction, contribution, and non-identifying display value. Never include identifiers, names, contact fields, or raw row payloads.

```python
records = []
for row_values, transformed in zip(values.values, Xt, strict=True):
    top = np.argsort(np.abs(row_values))[-top_k:][::-1]
    records.append(
        [
            {
                "column": str(column_names[i]),
                "direction": "increase" if row_values[i] > 0 else "decrease",
                "contribution": float(row_values[i]),
                "value": float(transformed[i]),
            }
            for i in top
        ]
    )
assert all(item["column"] not in protected for record in records for item in record)
```

5. Repeat explanations over at least five fixed bootstrap samples. Calculate top-k Jaccard overlap and Spearman rank correlation; flag fields below configured stability or with suspicious availability/timing.

```python
from scipy.stats import spearmanr

rank_corr = float(spearmanr(run_a["mean_abs_shap"], run_b["mean_abs_shap"]).statistic)
jaccard = len(set(top_a) & set(top_b)) / len(set(top_a) | set(top_b))
stable = rank_corr >= min_rank_correlation and jaccard >= min_topk_jaccard
```

6. Treat identifiers, post-outcome fields, near-perfect single-field effects, and unexplained disagreement as leakage candidates. Stop model use and hand suspicious fields to `detect-data-leakage`; do not remove fields silently.

7. Log exact MLflow fields: params `seed`, `data_version`, `model_run_id`, `evaluation_run_id`, `explainer_type`, `background_size`, `sample_size`, `permutation_repeats`, `top_k`, `bootstrap_runs`, `min_rank_correlation`, `min_topk_jaccard`; metrics `permutation_top1_pr_auc_drop`, `shap_topk_jaccard_mean`, `shap_rank_correlation_mean`, `unstable_field_count`, `leakage_candidate_count`; artifacts all CSV/JSONL files, global SHAP plot, permutation plot, lineage map, and leakage review.

```python
import mlflow

with mlflow.start_run(run_name="interpretability-audit"):
    mlflow.log_params(
        {
            "seed": 42,
            "data_version": data_version,
            "model_run_id": model_run_id,
            "evaluation_run_id": evaluation_run_id,
            "explainer_type": explainer_type,
            "framework": "pytorch"
            if model_family == "ann"
            else ("xgboost" if model_family == "xgboost" else "scikit-learn"),
            "background_size": len(training_background),
            "sample_size": len(sample),
            "permutation_repeats": 20,
            "top_k": top_k,
            "bootstrap_runs": 5,
            "min_rank_correlation": min_rank_correlation,
            "min_topk_jaccard": min_topk_jaccard,
        }
    )
    mlflow.log_metrics(
        {
            "permutation_top1_pr_auc_drop": float(
                permutation_df.mean_pr_auc_drop.iloc[0]
            ),
            "shap_rank_correlation_mean": rank_corr,
            "shap_topk_jaccard_mean": jaccard,
            "unstable_field_count": unstable_count,
            "leakage_candidate_count": leakage_count,
        }
    )
    mlflow.log_artifacts(artifact_dir, artifact_path="interpretability")
```

8. Handoff: consume the sealed model from `model-evaluation`; send cleared explanation artifacts to `model-management` and `model-deployment`, or route leakage findings back to data preparation and `model-training`.

## Validation checks and best practices

- [ ] Assert explanation inputs, outputs, plots, and logs contain neither `customerID` nor other PII.
- [ ] Verify permutation importance uses validation PR-AUC, 20 repeats, and a fixed seed.
- [ ] Verify SHAP array width equals transformed column-name count and row count equals sample size.
- [ ] Verify ANN explanations call a `torch.nn.Module` adapter on CPU and use a bounded background/sample size.
- [ ] Verify local contributions reconstruct the explainer output within `1e-4` where additivity applies.
- [ ] Verify at least five stability runs and record mean top-k Jaccard and rank correlation.
- [ ] Verify every leakage candidate has timing, lineage, and prediction-time availability review.
- [ ] Verify all artifacts exist and link to model/evaluation run IDs.
- Pitfall: explaining encoded columns without lineage; fix by persist `get_feature_names_out()` mappings.
- Pitfall: using test rows to choose explanations; fix by use configured validation samples.
- Pitfall: treating SHAP as causality; fix by label outputs as model associations.
- Hard rules: upstream split/fitting must be leakage-safe, stratified, seeded, and exclude `Churn`/`customerID`.
- Hard rules: never tune from test or substitute accuracy for ROC-AUC, PR-AUC, precision, and recall.
- Hard rules: never delete or overwrite data, DVC/MLflow artifacts, models, or migrations.
- Hard rules: stop for approval before threshold changes, promotion, deployment, push, or cost; never read `.env` or send PII/customer IDs to an LLM.
