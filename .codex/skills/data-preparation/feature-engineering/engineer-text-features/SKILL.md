---
name: engineer-text-features
description: Create privacy-aware lexical or vector text features with train-fitted vocabularies and explicit leakage screening.
---

# Engineer Text Features

## Purpose and use cases

Represent authorized free text for classical ML. Use only when text is available at prediction time and privacy/licensing allow it.

## Inputs and outputs

Inputs: training text, language/domain, PII policy, vocabulary limits, model family. Outputs: fitted vectorizer/transformer, sparse matrix, feature names, coverage/OOV report, and privacy/leakage audit.

## Execution and decision logic

1. Screen for target leakage, post-outcome notes, identifiers, and prohibited PII before tokenization.
2. Apply minimal domain-appropriate normalization; avoid destroying negation or meaningful casing blindly.
3. Prefer TF-IDF word/character n-grams for transparent baselines; hashing for bounded streaming memory; embeddings only with approved model/data handling.
4. Fit vocabulary/IDF on training folds only and cap dimensions/minimum document frequency.
5. Keep sparse outputs and combine through `ColumnTransformer`.

## Python example

```python
from sklearn.feature_extraction.text import TfidfVectorizer


def text_vectorizer() -> TfidfVectorizer:
    """Build a bounded, train-fitted text representation."""
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.98,
        max_features=50_000,
        sublinear_tf=True,
        dtype="float32",
    )
```

## Validation checks and best practices

Verify no raw text/PII is logged, vocabulary excludes target/identifier tokens, matrix is finite, and empty/OOV documents transform safely. Record vocabulary size and sparsity, test serving parity, and assess subgroup performance for language coverage.
