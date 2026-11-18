---
name: bivariate-analysis
description: Examine type-appropriate relationships between two variables or a feature and target while controlling leakage and multiplicity.
---

# Bivariate Analysis

## Purpose and use cases

Study association, separation, nonlinearity, and subgroup effects on training data. Use to form feature hypotheses, not to prove causality.

## Inputs and outputs

Inputs: two columns or feature/target, semantic types, sample weights/groups, prediction contract. Outputs: plot, effect/association estimate with uncertainty, sample counts, and caveats.

## Execution and decision logic

1. Verify both variables existed at prediction time and exclude IDs/proxies.
2. Select analysis: numeric-numeric scatter/hexbin and rank correlation; numeric-category box/ECDF and robust effect; category-category contingency and Cramer's V.
3. Show missingness as an explicit group and report group sizes.
4. Stratify by relevant time/group confounders when specified.
5. Correct or clearly label exploratory multiple testing. Prefer effect size and confidence intervals over p-values alone; do not infer causation.

## Python example

```python
import pandas as pd
from scipy.stats import spearmanr


def numeric_association(left: pd.Series, right: pd.Series) -> dict[str, float]:
    """Return pairwise-complete Spearman association."""
    pair = pd.concat([left, right], axis=1).dropna()
    result = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return {
        "n": float(len(pair)),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }
```

## Validation checks and best practices

Report pairwise sample size, missing exclusions, weights, and correction method. Check whether a few entities dominate. Use seeded downsampling for plots only; compute estimates on full affordable data. Validate suspicious relationships with time/group splits before acting.
