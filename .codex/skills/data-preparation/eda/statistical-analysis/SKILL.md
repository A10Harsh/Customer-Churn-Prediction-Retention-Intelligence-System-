---
name: statistical-analysis
description: Estimate uncertainty and run assumption-aware statistical tests for pre-specified data and feature hypotheses.
---

# Statistical Analysis

## Purpose and use cases

Quantify uncertainty in means, proportions, distributions, or group differences. Use when a decision needs more than descriptive EDA.

## Inputs and outputs

Inputs: hypothesis, estimand, variables, groups/time structure, sampling design, alpha and correction plan. Outputs: estimate, effect size, confidence interval, test result, assumptions, sample counts, and decision caveat.

## Execution and decision logic

1. State null/alternative, estimand, and analysis unit before testing.
2. Inspect independence, pairing, skew, variance, sparsity, and repeated entities.
3. Choose parametric tests only when assumptions are defensible; otherwise use rank, permutation, exact, or bootstrap methods.
4. Adjust for planned multiple comparisons and report practical effect sizes.
5. Separate confirmatory tests from exploratory findings; do not use test-set labels to select features.

## Python example

```python
import numpy as np


def bootstrap_mean_difference(
    a: np.ndarray, b: np.ndarray, *, seed: int = 42, draws: int = 5000
) -> tuple[float, tuple[float, float]]:
    """Estimate a mean difference and percentile bootstrap interval."""
    rng = np.random.default_rng(seed)
    diffs = np.array(
        [
            rng.choice(a, len(a), replace=True).mean()
            - rng.choice(b, len(b), replace=True).mean()
            for _ in range(draws)
        ]
    )
    return float(a.mean() - b.mean()), tuple(np.quantile(diffs, [0.025, 0.975]))
```

## Validation checks and best practices

Validate analysis unit, missing-data treatment, minimum group sizes, and reproducibility. Report exact method and seed. Avoid p-value fishing, normality testing as a ritual, and interpreting statistical significance as business importance or causality.
