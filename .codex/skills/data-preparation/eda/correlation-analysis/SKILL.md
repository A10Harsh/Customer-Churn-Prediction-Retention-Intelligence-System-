---
name: correlation-analysis
description: Compute type-appropriate association matrices and redundancy candidates with missingness and nonlinear limitations made explicit.
---

# Correlation Analysis

## Purpose and use cases

Identify redundant numeric features, monotonic relationships, and categorical associations. Use for diagnostics and feature review, not automatic deletion.

## Inputs and outputs

Inputs: training frame, semantic types, association method, minimum pair count. Outputs: association matrix, pair counts, high-association candidates, heatmap, and rationale for any action.

## Execution and decision logic

1. Exclude IDs, target, and leakage candidates.
2. Use Pearson for approximately linear numeric relations, Spearman/Kendall for monotonic or ordinal data, point-biserial for binary-numeric, and Cramer's V or mutual information for categorical pairs.
3. Produce pairwise sample counts and inspect missingness patterns.
4. Review highly associated features for duplicate meaning, stability, cost, and availability.
5. Drop or combine only with domain and cross-validation evidence; correlation misses nonlinear dependence and is not causation.

## Python example

```python
import pandas as pd


def numeric_correlations(
    frame: pd.DataFrame, method: str = "spearman"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return numeric associations and pairwise non-null counts."""
    numeric = frame.select_dtypes(include="number")
    present = numeric.notna().astype(int)
    return numeric.corr(method=method), present.T @ present
```

## Validation checks and best practices

Assert symmetry, unit diagonal where defined, and adequate pair counts. Compare Pearson and rank results when tails matter. Cluster heatmaps for readability, mask duplicate triangles, never fill missing values solely to calculate correlation, and record thresholds in configuration.
