---
name: multivariate-analysis
description: Explore conditional structure, interactions, clusters, and redundancy across multiple training features without leaking evaluation outcomes.
---

# Multivariate Analysis

## Purpose and use cases

Investigate structure not visible pairwise: conditional target patterns, interacting features, multicollinearity, and segments. Use after univariate/bivariate checks on a curated feature subset.

## Inputs and outputs

Inputs: training features, optional target/group/time, schema, sampling and seed configuration. Outputs: interaction/redundancy candidates, visualizations, cluster or projection diagnostics, and limitations.

## Execution and decision logic

1. Remove targets, IDs, leakage candidates, and unsupported PII; preprocess within an exploratory pipeline.
2. Select variables by domain question, not convenience.
3. Use faceting/partial summaries for a few variables; PCA/SVD or clustering only for exploration and with scaling suited to inputs.
4. Evaluate stability across seeded resamples or folds.
5. Treat clusters and projections as hypotheses; do not convert them into production features without validation.

## Python example

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def exploratory_projection(
    frame: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, PCA]:
    """Fit a two-component exploratory projection on training numeric data."""
    pipe = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=seed))
    values = pipe.fit_transform(frame)
    pca = pipe.named_steps["pca"]
    return pd.DataFrame(values, index=frame.index, columns=["pc1", "pc2"]), pca
```

## Validation checks and best practices

Report explained variance/cluster stability and preprocessing, verify no target entered unsupervised transforms, and check results across samples. Avoid unreadable pair plots, causal claims, and projections fitted on combined splits. Keep original feature interpretability alongside reduced views.
