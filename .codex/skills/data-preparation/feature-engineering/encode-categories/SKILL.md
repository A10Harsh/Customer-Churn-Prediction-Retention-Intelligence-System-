---
name: encode-categories
description: Select and implement one-hot, ordinal, frequency, hashing, or cross-fitted target encoding based on cardinality and model needs.
---

# Encode Categories

## Purpose and use cases

Convert categorical features to model inputs without imposing false order or leaking target statistics.

## Inputs and outputs

Inputs: train splits/folds, semantic types, cardinality, known order, model family, memory budget. Outputs: fitted encoder pipeline, feature names, category mappings, unseen-level behavior, and diagnostics.

## Execution and decision logic

- Use one-hot for nominal low/moderate cardinality and `handle_unknown="ignore"`.
- Use ordinal encoding only for a real ordered scale with explicit category order.
- Use frequency or hashing for high cardinality when acceptable; frequency mappings fit on train only.
- Use target encoding only when incremental value justifies it, with out-of-fold training values, smoothing, and train-only mappings for inference.
- Never integer-label nominal values as if ordered; never encode identifiers as features.

## Python example

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def choose_encoder(
    *, ordered_categories: list[list[str]] | None, min_frequency: int = 10
) -> object:
    """Choose an encoder without learning from evaluation data."""
    if ordered_categories:
        return OrdinalEncoder(
            categories=ordered_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
        sparse_output=True,
    )
```

## Validation checks and best practices

Transform unseen/null levels, confirm stable output order/names, and ensure mappings contain training categories only. Check sparse dimensionality and collision/rare rates. Persist encoder with model; test serialization and train/serve parity.
