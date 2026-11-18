---
name: build-preprocessing-pipeline
description: Assemble typed cleaning and feature transforms into a serializable scikit-learn pipeline shared by training, batch scoring, and serving.
---

# Build Preprocessing Pipeline

## Purpose and use cases

Eliminate train/serve skew and leakage by making every learned transform part of one fitted object.

## Inputs and outputs

Inputs: approved schema/feature lists, cleaning and encoding choices, estimator/sampler, seed, config. Outputs: unfitted pipeline definition, fitted pipeline from training only, feature names, serialized artifact metadata, and transform validation report.

## Execution and decision logic

1. Separate deterministic row-level validation from train-fitted transformations.
2. Build numeric/categorical/text branches with explicit column lists or schema selectors.
3. Order numeric steps: impute, optional transform, scale; categorical: impute/group, encode.
4. Combine with `ColumnTransformer`; use imbalanced-learn Pipeline when sampling; append selector/reducer/model as required.
5. Fit once on training or inside CV; call only `transform`/`predict` elsewhere. Pin schema and artifact versions.

## Python example

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    """Build a reusable typed preprocessor."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, num), ("categorical", categorical, cat)],
        remainder="drop",
        verbose_feature_names_out=True,
    )
```

## Validation checks and best practices

Assert required input columns, stable output names/order, finite matrices, unseen-category behavior, and no fitting on validation/test. Test serialization round trip and batch-versus-single-row parity. Persist config, code/data version, seed, and feature lineage with the artifact.
