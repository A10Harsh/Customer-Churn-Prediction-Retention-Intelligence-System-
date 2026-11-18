---
name: reduce-dimensionality
description: Apply PCA, TruncatedSVD, or other compression only when validated memory, noise, or collinearity benefits justify reduced interpretability.
---

# Reduce Dimensionality

## Purpose and use cases

Compress high-dimensional numeric or sparse representations for compatible estimators. Use after simpler selection alternatives are considered.

## Inputs and outputs

Inputs: train-fitted preprocessing output, sparsity, model family, variance/component budget, CV design. Outputs: fitted reducer in pipeline, component matrix, explained-variance or reconstruction diagnostics, names, and tradeoff report.

## Execution and decision logic

- Use PCA for centered dense numeric data; scale first when units differ.
- Use TruncatedSVD for sparse one-hot/text matrices without centering.
- Choose component count inside training CV by performance/latency/variance goal, never on the test set.
- Avoid reduction when per-feature explanations, monotonic constraints, or simple sparse linear models matter more.
- Fit reducer on training folds only and serialize it with preprocessing.

## Python example

```python
from sklearn.decomposition import PCA, TruncatedSVD


def reducer(*, sparse: bool, components: int, seed: int = 42) -> object:
    """Return a deterministic unfitted dense or sparse reducer."""
    if sparse:
        return TruncatedSVD(n_components=components, random_state=seed)
    return PCA(n_components=components, svd_solver="auto", random_state=seed)
```

## Validation checks and best practices

Ensure component count is below valid rank, transformed matrices are finite, and train/serve shapes match. Report explained variance where defined, reconstruction/stability, runtime, and CV delta versus no reduction. Retain component loadings and input feature names for audit.
