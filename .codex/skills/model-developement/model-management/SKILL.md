---
name: model-management
description: Track, compare, register, promote, and roll back churn models in MLflow when governed model lifecycle actions are requested.
---

## Purpose and use cases

Maintain auditable model lineage and registry state after training and evaluation. Compare immutable runs, register qualified versions, and prepare promotion or rollback plans while stopping for human approval before changing aliases or serving state.

- Standardize MLflow experiment and run naming.
- Register an evaluated candidate as a non-serving version.
- Compare a challenger with the current champion.
- Promote or roll back an alias after explicit approval.
- Do not train or tune models; use `model-training` or `model-optimization`.
- Do not execute a serving rollout; use `model-deployment`.

## Inputs and outputs

| Name | Type | Required | Description |
|---|---|---:|---|
| Training/evaluation run IDs | strings | Yes | Immutable MLflow runs with required lineage and metric suite. |
| Model artifact URI | MLflow URI | Yes | Logged sklearn Pipeline including preprocessing. |
| Registry config | YAML/dict | Yes | Experiment `churn-retention`, registered name `churn-classifier`, aliases, comparison gates. |
| Data/model versions | strings | Yes | DVC data hash, git SHA, artifact digest, and schema version. |
| Approval record | ticket/string | Conditional | Required for champion alias change, rollback, deployment, push, or cost. |

Outputs:

- MLflow model version under `churn-classifier` with model-family tags and description.
- `artifacts/model_management/comparison-<id>.json`: challenger/champion evidence.
- `artifacts/model_management/promotion-plan-<id>.md` or `rollback-plan-<id>.md`.
- Approval-linked registry event; no alias mutation without approval.

## Execution

1. Configure the tracking URI from runtime configuration without reading `.env`. Select the fixed experiment and verify every candidate run is `FINISHED`.

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("churn-retention")
client = MlflowClient()
run = client.get_run(candidate_run_id)
assert run.info.status == "FINISHED"
```

2. Validate required lineage, metrics, and artifacts before registration. Reject runs lacking the one-time test report or full metric suite.

```python
required_params = {"seed", "data_version", "decision_threshold"}
required_metrics = {"test_roc_auc", "test_pr_auc", "test_precision", "test_recall"}
assert required_params <= run.data.params.keys()
assert required_metrics <= run.data.metrics.keys()
artifacts = {a.path for a in client.list_artifacts(candidate_run_id)}
assert "model" in artifacts and "evaluation" in artifacts
```

3. Register a new immutable version and tag it; registration alone must not set the champion alias.

```python
result = mlflow.register_model(f"runs:/{candidate_run_id}/model", "churn-classifier")
for key, value in {
    "data_version": run.data.params["data_version"],
    "git_sha": git_sha,
    "evaluation_run_id": evaluation_run_id,
    "schema_version": schema_version,
}.items():
    client.set_model_version_tag("churn-classifier", result.version, key, str(value))
client.set_model_version_tag(
    "churn-classifier", result.version, "model_family", model_family
)
client.set_model_version_tag(
    "churn-classifier", result.version, "serving_eligible", "true"
)
client.set_model_version_tag(
    "churn-classifier", result.version, "framework", run.data.params["framework"]
)
```

4. Compare challenger and champion using the sealed test report plus operational constraints. Apply identical serving gates to logistic regression, Random Forest, XGBoost, and PyTorch ANN; do not reevaluate test, retune, or silently change gates.

```python
champion = client.get_model_version_by_alias("churn-classifier", "champion")
champion_run = client.get_run(champion.run_id)
comparison = {
    m: run.data.metrics[m] - champion_run.data.metrics[m]
    for m in ("test_pr_auc", "test_roc_auc", "test_precision", "test_recall")
}
eligible = (
    comparison["test_pr_auc"] >= min_pr_auc_delta and latency_ms <= max_latency_ms
)
```

5. Log the comparison in a separate governance run. Exact params: `model_family`, `serving_eligible`, `baseline_run_id`, candidate/champion IDs and versions, data/git/schema versions, threshold, and gate config; metrics: candidate/champion/delta for ROC-AUC, PR-AUC, precision, recall, Brier, plus latency and artifact size; artifacts: comparison JSON, evaluation reports, signatures, approval request.

```python
with mlflow.start_run(run_name="registry-comparison"):
    mlflow.log_params(
        {
            "model_family": model_family,
            "serving_eligible": True,
            "baseline_run_id": baseline_run_id,
            "candidate_run_id": candidate_run_id,
            "champion_run_id": champion.run_id,
            "candidate_version": result.version,
            "champion_version": champion.version,
            "data_version": run.data.params["data_version"],
            "git_sha": git_sha,
            "schema_version": schema_version,
            "decision_threshold": run.data.params["decision_threshold"],
            "gate_config_version": gate_config_version,
        }
    )
    mlflow.log_metrics(
        {f"delta_{k.removeprefix('test_')}": float(v) for k, v in comparison.items()}
    )
    mlflow.log_artifact(comparison_path, artifact_path="model_management")
```

6. Prepare the promotion plan. If champion promotion is requested, stop and ask for explicit approval naming the registered version, evidence, threshold, target alias, and rollback version.

```python
if not approval_id:
    raise RuntimeError(
        f"Approval required to promote churn-classifier version {result.version}"
    )
client.set_registered_model_alias("churn-classifier", "champion", result.version)
client.set_model_version_tag(
    "churn-classifier", result.version, "approval_id", approval_id
)
```

7. For rollback, identify the prior immutable version, compare schema compatibility, prepare the action, and stop for explicit approval before moving the alias. Never delete the failed version or its artifacts.

8. Handoff: receive evidence from `model-evaluation` and `model-interpretability`; after approved alias selection, provide the exact model URI to `model-deployment`, which then hands runtime evidence to `model-monitoring`.

## Validation checks and best practices

- [ ] Verify experiment is `churn-retention`, registered model is `churn-classifier`, and model family is tagged.
- [ ] Verify candidate run is finished and has seed, data version, threshold, signature, model, and final report.
- [ ] Verify test ROC-AUC, PR-AUC, precision, and recall are finite real-run values.
- [ ] Verify each registry version has data, git, evaluation-run, schema, framework, and `serving_eligible=true` tags.
- [ ] Verify comparison JSON exists and includes challenger, champion, deltas, gates, and artifact hashes.
- [ ] Assert every candidate, including PyTorch ANN, is tagged `serving_eligible=true`; champion alias changes still require approval and all gates.
- [ ] Assert champion alias changes have a non-empty approval ID and recorded rollback version.
- [ ] Verify rollback changes only an alias and preserves all versions and runs.
- Pitfall: using mutable stage names as lineage; fix by record exact run and model-version IDs.
- Pitfall: reusing test to justify promotion; fix by compare sealed reports without reevaluation.
- Pitfall: deleting a bad version; fix by retain it and tag its disposition.
- Hard rules: upstream training must split first, fit preprocessing on training only, exclude target/ID, stratify, and fix seeds.
- Hard rules: never tune from test or report accuracy alone; require the full metric suite.
- Hard rules: never delete or overwrite data, DVC/MLflow artifacts, registry versions, or migrations.
- Hard rules: stop for approval before promotion, rollback alias mutation, threshold change, deployment, push, or cost; never read `.env` or expose PII/customer IDs to an LLM.
