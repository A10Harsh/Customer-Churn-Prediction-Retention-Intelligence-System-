---
name: transform-distributions
description: Select log, power, quantile, clipping, or identity transforms from domain support, model needs, and train-only diagnostics.
---

# Transform Distributions

## Purpose and use cases

Reduce problematic skew, heteroscedasticity, or leverage for sensitive models while preserving meaning where possible.

## Inputs and outputs

Inputs: training numeric features, domain bounds, distribution analysis, estimator family. Outputs: fitted transform pipeline, parameters, transformed feature names, inverse-transform status, and sensitivity report.

## Execution and decision logic

1. Start with identity; trees often need no distribution transform.
2. Use `log1p` for nonnegative multiplicative quantities; use signed/asinh only with a defensible meaning.
3. Use Yeo-Johnson for mixed signs, Box-Cox for strictly positive values, quantile transforms only when rank mapping and tail saturation are acceptable.
4. Clip only against domain rules or train-fitted robust limits and retain outlier indicators when useful.
5. Fit learned transforms on training folds and validate downstream benefit.

## Python example

```python
import numpy as np
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    QuantileTransformer,
)


def select_transform(kind: str, seed: int = 42) -> object:
    """Return an unfitted distribution transformer."""
    if kind == "log1p":
        return FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    if kind == "yeo-johnson":
        return PowerTransformer(method="yeo-johnson")
    if kind == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=seed)
    return "passthrough"
```

## Validation checks and best practices

Validate transform domain, finite outputs, monotonicity where expected, and stable inverse transform when promised. Test zeros/negatives/constants/extremes. Record parameters and do not choose a transform solely to make a histogram appear normal.
