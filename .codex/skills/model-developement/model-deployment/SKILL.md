---
name: model-deployment
description: Package and release an approved churn model when schema validation, smoke tests, canary evidence, and a rollback plan are required.
---

## Purpose and use cases

Prepare a registry-pinned model for safe FastAPI serving with strict request schemas and reproducible packaging. Validate locally, create a canary and rollback plan, and stop for explicit approval before any deployment, traffic change, remote push, or cost-incurring action.

- Package an approved MLflow model version into the API image.
- Add request validation and batch-size controls.
- Run local container and endpoint smoke tests.
- Prepare canary gates and rollback commands for review.
- Do not register or promote a model; use `model-management`.
- Do not assess post-release drift; use `model-monitoring`.

## Inputs and outputs

| Name | Type | Required | Description |
|---|---|---:|---|
| Approved model URI | `models:/...@alias` or version URI | Yes | Exact immutable `churn-classifier` version plus approval record. |
| Model signature | MLflow signature/schema | Yes | Expected input names, types, nullability, and output shape. |
| API code/config | FastAPI/Pydantic/YAML | Yes | Endpoints, limits, threshold, timeouts, health checks; no secrets. |
| Container definition | Dockerfile/compose | Yes | Python 3.11 runtime with pinned dependencies. |
| Evaluation/interpretability artifacts | MLflow URIs | Yes | Sealed metrics, threshold, lineage, and explanation contract. |
| Deployment approval | record/string | Yes for mutation | Required before deploy, canary traffic, rollback, remote push, or cost. |

Outputs:

- Immutable image tagged with model version and Git SHA; local builds may use an equivalent digest.
- `artifacts/deployment/<release_id>/manifest.json`, `smoke-results.json`, and `rollback-plan.md`.
- MLflow release-validation run linked to model version and approval ID.
- Approved deployment/canary event, or a prepared plan with no external mutation.

## Execution

1. Resolve the exact registered version and verify champion/promotion approval, evaluation report, threshold, signature, and schema compatibility. Never deploy from a mutable local file.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri=tracking_uri)
mv = client.get_model_version("churn-classifier", model_version)
assert mv.tags["approval_id"] == promotion_approval_id
assert mv.tags["schema_version"] == api_schema_version
assert mv.tags["serving_eligible"] == "true"
model_uri = f"models:/churn-classifier/{model_version}"
```

2. Enforce request validation before inference. Exclude target and identifier fields explicitly and reject unknown fields.

```python
from pydantic import BaseModel, ConfigDict, Field, model_validator

protected = {"Churn", "customerID"}


class ChurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenure: int = Field(ge=0)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)
    Contract: str

    @model_validator(mode="before")
    @classmethod
    def reject_protected(cls, values: dict) -> dict:
        assert protected.isdisjoint(values)
        return values
```

3. Load the pinned MLflow Pipeline once at startup and apply its approved threshold. Log request IDs and model metadata only; never raw payloads, identifiers, or PII.

```python
import mlflow.sklearn

model = mlflow.sklearn.load_model(model_uri)


def predict_one(request: ChurnRequest) -> dict[str, float | bool]:
    frame = __import__("pandas").DataFrame([request.model_dump()])
    probability = float(model.predict_proba(frame)[0, 1])
    return {
        "churn_probability": probability,
        "will_churn": probability >= approved_threshold,
    }
```

4. Build locally with immutable labels and scan the resulting metadata. For an ANN release, pin the CPU PyTorch runtime and verify the adapter plus state-dict digest are included. Do not push the image or contact a paid registry without explicit approval.

```bash
docker build --label model.version="$MODEL_VERSION" --label org.opencontainers.image.revision="$GIT_SHA" -t churn-api:"$RELEASE_ID" -f deployment/Dockerfile .
docker image inspect churn-api:"$RELEASE_ID" --format '{{.Id}}'
```

5. Run smoke tests against the local container: health, one valid prediction, invalid target/identifier payloads, missing required field, unknown category policy, batch limit, deterministic response, and model/version headers.

```python
from fastapi.testclient import TestClient

client = TestClient(app)
assert client.get("/health").status_code == 200
ok = client.post("/predict", json=valid_payload)
assert ok.status_code == 200 and 0 <= ok.json()["churn_probability"] <= 1
for key in protected:
    assert (
        client.post("/predict", json={**valid_payload, key: "forbidden"}).status_code
        == 422
    )
```

6. Write a canary plan with traffic fraction, duration, minimum requests, error-rate/latency/probability-distribution gates, owner, and exact rollback target. Stop and ask for approval before deployment or traffic mutation.

```python
if not deployment_approval_id:
    raise RuntimeError(
        f"Approval required to deploy release {release_id} with model {model_version}"
    )
```

7. After approval, execute only the reviewed command/provider plan. Observe canary gates; roll back to the named previous release only with explicit rollback approval unless the approved plan already authorizes an automatic safety rollback. Preserve both releases and all artifacts.

8. Log exact MLflow fields: params `release_id`, `model_name`, `model_family`, `serving_eligible`, `model_version`, `model_run_id`, image/git/data/schema versions, threshold, deployment/canary/rollback fields, and approval IDs; metrics smoke results, local latency and, after approved canary, error rate, latency, and request count; artifacts manifest, image metadata/SBOM if available, smoke results, schema, canary plan, and rollback plan.

```python
import mlflow

with mlflow.start_run(run_name="release-validation"):
    mlflow.log_params(
        {
            "release_id": release_id,
            "model_name": "churn-classifier",
            "model_family": mv.tags["model_family"],
            "serving_eligible": mv.tags["serving_eligible"],
            "model_version": model_version,
            "model_run_id": mv.run_id,
            "image_digest": image_digest,
            "git_sha": git_sha,
            "data_version": data_version,
            "schema_version": api_schema_version,
            "decision_threshold": approved_threshold,
            "deployment_target": target,
            "canary_fraction": canary_fraction,
            "rollback_release_id": rollback_release_id,
            "promotion_approval_id": promotion_approval_id,
            "deployment_approval_id": deployment_approval_id,
        }
    )
    mlflow.log_metrics(smoke_metrics)
    mlflow.log_artifacts(release_dir, artifact_path="deployment")
```

9. Handoff: receive the approved version from `model-management`; after an approved release, provide deployment ID, model version, schema, threshold, and reference data version to `model-monitoring`.

## Validation checks and best practices

- [ ] Verify model URI resolves to an immutable version with promotion and deployment approval IDs.
- [ ] For ANN, verify the image contains the pinned PyTorch runtime, adapter, state-dict digest, and a CPU reload smoke test.
- [ ] Assert request schema forbids `Churn`, `customerID`, extra fields, invalid types, and oversized batches.
- [ ] Verify valid and invalid endpoint tests pass against the response schema.
- [ ] Verify the loaded pipeline includes preprocessing and matches the evaluation artifact digest/signature.
- [ ] Verify smoke failures equal `0`, probabilities lie in `[0, 1]`, and repeats match within `1e-6` for ANN or `1e-12` otherwise.
- [ ] Verify manifest records image digest, Git SHA, model/data/schema versions, threshold, and rollback release.
- [ ] Verify canary gates and rollback plan exist before any external mutation.
- Pitfall: packaging a local unregistered model; fix by pin the approved registry version URI.
- Pitfall: accepting identifiers through `extra="ignore"`; fix by forbid extras and protected keys.
- Pitfall: deploying immediately after build; fix by complete local smoke evidence and request approval.
- Hard rules: deployment consumes a model trained with split-first, train-only preprocessing, target/ID exclusion, stratification, and fixed seeds.
- Hard rules: never retune from test or serving data; preserve real ROC-AUC, PR-AUC, precision, and recall evidence.
- Hard rules: never delete or overwrite data, DVC/MLflow artifacts, model versions, releases, or migrations.
- Hard rules: stop for explicit approval before deployment, canary traffic, rollback mutation, promotion, threshold change, remote push, or cost; never read `.env` or expose PII/customer IDs to an LLM.
