---
name: mlops
description: Build reproducible DVC and CI workflows for churn modeling when data, code, environment, and model lineage must be automated and verified.
---

## Purpose and use cases

Connect data preparation, training, evaluation, and packaging through reproducible local pipelines and CI gates. Pin environments, declare DVC stage dependencies, and publish lineage manifests without performing approval-gated promotion or deployment.

- Create or update `dvc.yaml` for train/evaluate stages.
- Add Ruff, pytest, and reproducibility checks to CI.
- Pin Python and dependency versions for repeatable runs.
- Trace a model artifact to data, code, config, and MLflow runs.
- Do not manage registry aliases; use `model-management`.
- Do not deploy or monitor a live model; use `model-deployment` or `model-monitoring`.

## Inputs and outputs

| Name | Type | Required | Description |
|---|---|---:|---|
| Stage entry points | Python modules | Yes | Deterministic train/evaluate commands under `src/` or `pipelines/`. |
| Data artifacts | DVC-tracked paths | Yes | Immutable split inputs; raw data remains outside Git. |
| Configs | YAML files | Yes | Seeds, paths, model settings, costs, and gates; no secrets. |
| Environment spec | `pyproject.toml`/lock file | Yes | Python 3.11 and exact resolved dependencies. |
| Test suite | pytest files | Yes | Leakage, parity, schema, and pipeline tests. |
| MLflow run IDs | strings | Yes | Training/evaluation lineage produced upstream. |

Outputs:

- `dvc.yaml` and `dvc.lock` with declared dependencies, parameters, metrics, and outputs.
- `.github/workflows/ci.yml` with Ruff, pytest, and DVC reproduction checks.
- `artifacts/lineage/<model_run_id>.json` linking data, code, config, environment, and model digests.
- Reproducibility report in MLflow under `mlops/`.

## Execution

1. Inventory existing DVC-tracked data and stage entry points. Never add raw CSV contents or secrets to Git; add only `.dvc`, lock, config, and code metadata.

```bash
dvc status
dvc list . data/processed --dvc-only
git status --short
```

2. Define stages with explicit dependencies and new, non-overwriting output directories. Ensure the split stage precedes every fitting stage and all config values are tracked as dependencies or parameters.

```yaml
vars:
  - configs/run.yaml  # contains a new immutable run_id
stages:
  train:
    cmd: python -m src.models.train --config configs/model.yaml --run-id ${run_id}
    deps: [src/models/train.py, src/features, data/processed/train.parquet, data/processed/validation.parquet, configs/model.yaml, configs/run.yaml]
    params: [configs/model.yaml:seed, configs/model.yaml:model]
    outs: [artifacts/training/${run_id}]
    metrics: [artifacts/training/${run_id}/metrics.json]
  evaluate:
    cmd: python -m src.models.evaluate --config configs/evaluation.yaml --run-id ${run_id}
    deps: [src/models/evaluate.py, artifacts/training/${run_id}, data/processed/test.parquet, configs/evaluation.yaml]
    outs: [artifacts/evaluation/${run_id}]
    metrics: [artifacts/evaluation/${run_id}/test_metrics.json]
```

3. Set a new `run_id` whenever dependencies change, assert its output paths do not exist, reproduce locally, and inspect changes before accepting a lock update. Do not force, delete caches, or overwrite protected artifacts.

```bash
test ! -e "artifacts/training/$RUN_ID"
test ! -e "artifacts/evaluation/$RUN_ID"
dvc repro
dvc status
dvc diff --show-hash
```

4. Add CI gates using pinned Python 3.11 and the project extras. CI uses fake external clients and no real secret or paid API.

```yaml
- uses: actions/setup-python@v5
  with: {python-version: "3.11", cache: pip}
- run: python -m pip install -e '.[dev]'
- run: ruff check .
- run: ruff format --check .
- run: pytest
- run: dvc repro --dry
```

5. Generate a lineage manifest from observable hashes and upstream run IDs. Use DVC/Git/MLflow metadata; never inspect `.env`.

```python
import hashlib, json, subprocess
from pathlib import Path

manifest = {
    "model_run_id": model_run_id,
    "evaluation_run_id": evaluation_run_id,
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "dvc_lock_sha256": hashlib.sha256(Path("dvc.lock").read_bytes()).hexdigest(),
    "python": "3.11",
    "seed": 42,
    "data_version": data_version,
}
out = Path("artifacts/lineage") / f"{model_run_id}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
```

6. Log exact fields to an MLflow reproducibility run: params `seed`, `data_version`, `git_sha`, `dvc_lock_sha256`, `python_version`, `dependency_lock_sha256`, `training_run_id`, `evaluation_run_id`, `pipeline_stage`; metrics `ruff_errors`, `pytest_failures`, `reproducibility_probability_max_abs_diff`, and required evaluation metrics copied with source run IDs; artifacts `dvc.yaml`, `dvc.lock`, environment lock, CI workflow, test report, and lineage JSON.

```python
import mlflow

with mlflow.start_run(run_name="reproducibility-check"):
    mlflow.log_params(
        {
            k: manifest[k]
            for k in ("seed", "data_version", "git_sha", "dvc_lock_sha256", "python")
        }
    )
    mlflow.log_metrics(
        {
            "ruff_errors": 0,
            "pytest_failures": 0,
            "reproducibility_probability_max_abs_diff": max_abs_diff,
        }
    )
    mlflow.log_artifact(out, artifact_path="mlops")
```

7. Rerun the same stage in an isolated new output directory and compare probability files by stable row key; require maximum absolute difference at or below `1e-12` unless the configured estimator documents a wider tolerance.

8. Handoff: consume split/pipeline artifacts from data-preparation skills, support `model-training` through `model-evaluation`, then provide lineage to `model-management` and `model-deployment`.

## Validation checks and best practices

- [ ] Verify `dvc dag` places split before train and train before evaluate.
- [ ] Verify every code/config/data input is in `deps` or `params` and every generated artifact is declared.
- [ ] Verify raw data, `.env`, credentials, and customer IDs are absent from Git diffs and logs.
- [ ] Verify `ruff check .`, `ruff format --check .`, and `pytest` exit `0`; no-tests-collected is inconclusive.
- [ ] Verify `dvc repro --dry` succeeds in CI and a local full reproduction has been run when data is available.
- [ ] Verify lineage JSON includes data, Git, DVC lock, dependency lock, seed, training run, and evaluation run IDs.
- [ ] Verify two isolated runs differ in probabilities by no more than the configured tolerance.
- Pitfall: untracked config changes; fix by declare config in `deps`/`params`.
- Pitfall: CI downloads private data or calls a real LLM; fix by use fixtures and fake clients.
- Pitfall: reusing a run ID overwrites artifacts; fix by create a new immutable run ID and fail if either output path exists.
- Hard rules: upstream stages split before fit, train-fit preprocessing inside Pipeline, exclude `Churn`/`customerID`, stratify, and fix seeds.
- Hard rules: selection uses validation/CV; test runs once; log ROC-AUC, PR-AUC, precision, and recall from real runs.
- Hard rules: never delete or overwrite data, DVC/MLflow artifacts, registered models, or migrations.
- Hard rules: stop for approval before threshold changes, promotion, deployment, push, or cost; never read `.env` or send PII/customer IDs to an LLM.
