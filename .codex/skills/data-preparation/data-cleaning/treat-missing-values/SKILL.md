---
name: treat-missing-values
description: Diagnose missingness mechanisms and implement train-fitted deletion, indicators, or imputation without leakage.
---

# Treat Missing Values

## Purpose and use cases

Choose a defensible missing-data policy by feature and model family. Use after quality assessment and splitting.

## Inputs and outputs

Inputs: split datasets, semantic schema, missingness profile, domain rules, model family. Outputs: documented policy, fitted imputation pipeline, optional indicators, and before/after validation report.

## Execution and decision logic

1. Distinguish structural, unavailable, invalid, and unexpectedly missing values.
2. Drop rows only when rare, plausibly unbiased, and target/key integrity remains; drop features only when value is insufficient and documented.
3. Use median for skewed numeric, most-frequent/constant for categorical, or iterative/KNN only with justified validation and cost.
4. Add indicators when missingness may carry stable information; beware operational changes.
5. Fit all imputers on training folds only and preserve null handling in the serving pipeline. Never impute the target.

## Python example

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def imputation_pipeline(
    numeric: list[str], categorical: list[str]
) -> ColumnTransformer:
    """Build train-fitted imputation by semantic type."""
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
```

## Validation checks and best practices

Assert unchanged row order/count unless deletion was approved, no unexpected null/non-finite outputs, and pipeline transforms validation/test without fitting. Compare imputed distributions and CV performance, log per-column policies, preserve missingness baselines, and test all-null or unseen-category edge cases.
