---
name: assess-model-readiness
description: Run the final evidence-based gate confirming data, splits, leakage controls, pipeline, reproducibility, and artifacts are ready for model training.
---

# Assess Model Readiness

## Purpose and use cases

Decide whether prepared data may enter model training. Use after all cleaning/feature/preprocessing work and before expensive tuning or registry runs.

## Inputs and outputs

Inputs: prediction contract, manifests, quality/leakage/schema reports, split assignments, fitted preprocessing pipeline, feature lineage, seed/config/version metadata. Outputs: pass/warn/fail checklist with evidence links, blocking issues, matrix summaries, and readiness manifest.

## Execution and decision logic

Gate on: valid target and unit of analysis; immutable/versioned sources; disjoint appropriate splits; no unresolved leakage; approved schema/quality tolerances; train-only fitted and serializable preprocessing; finite aligned model matrices; documented imbalance/CV/metrics plan; reproducible seed/config; and protected PII/artifacts. Any missing evidence is not a pass. Warnings need owner and acceptance; blockers stop training.

## Python example

```python
from collections.abc import Mapping

REQUIRED_GATES = (
    "target_valid",
    "splits_isolated",
    "leakage_clear",
    "schema_valid",
    "matrix_finite",
    "pipeline_serializable",
    "reproducible",
)


def readiness_status(checks: Mapping[str, bool]) -> tuple[str, list[str]]:
    """Return a fail-closed readiness decision and missing/failed gates."""
    blockers = [gate for gate in REQUIRED_GATES if checks.get(gate) is not True]
    return ("fail" if blockers else "pass"), blockers
```

## Validation checks and best practices

Execute—not merely assert—schema, overlap, finiteness, serialization round-trip, and deterministic rerun checks. Record actual counts/metrics and artifact versions. Require PR-AUC, ROC-AUC, precision, and recall plans for imbalanced classification; keep the test set untouched until final evaluation. Never waive leakage or target-integrity failures.
