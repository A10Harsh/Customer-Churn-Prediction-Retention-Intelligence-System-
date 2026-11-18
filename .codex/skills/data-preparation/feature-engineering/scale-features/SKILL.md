---
name: scale-features
description: Choose and fit standard, robust, min-max, or no scaling according to model geometry, tails, sparsity, and domain bounds.
---

# Scale Features

## Purpose and use cases

Align numeric scales for distance-, gradient-, or regularization-sensitive models. Skip unnecessary scaling for most tree models unless another pipeline step needs it.

## Inputs and outputs

Inputs: training numeric features, model family, distribution/outlier diagnostics, sparsity and bounds. Outputs: fitted scaler within pipeline, learned parameters, transformed features, and diagnostics.

## Execution and decision logic

- StandardScaler: roughly stable finite distributions; set `with_mean=False` for sparse matrices.
- RobustScaler: heavy tails or legitimate extremes.
- MinMaxScaler: bounded-input algorithms when deployment ranges are controlled; it does not handle future extremes robustly.
- No scaling: invariant tree models or already comparable meaningful units.
- Fit on training folds only after imputation; never scale target unless a target transformer is explicitly part of regression.

## Python example

```python
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def select_scaler(kind: str, *, sparse: bool = False) -> object:
    """Return an unfitted scaler selected by reviewed policy."""
    choices = {
        "standard": StandardScaler(with_mean=not sparse),
        "robust": RobustScaler(with_centering=not sparse),
        "minmax": MinMaxScaler(),
        "none": "passthrough",
    }
    return choices[kind]
```

## Validation checks and best practices

Assert finite outputs, expected dimensions, train-derived parameters, and safe handling of constants. Test serialization and inference extremes. Inspect—not require—training center/scale, and compare scaler choices through leakage-safe CV for the intended estimator.
