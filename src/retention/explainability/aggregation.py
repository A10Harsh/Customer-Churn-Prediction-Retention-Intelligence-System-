"""Map fitted ColumnTransformer outputs to the original customer predictors."""

import numpy as np


def original_feature_map(preprocessor, expected: tuple[str, ...]) -> tuple[str, ...]:
    """Use fitted column slices and encoder category counts, never name parsing."""
    columns = preprocessor.named_steps["columns"]
    encoded_names = tuple(columns.get_feature_names_out())
    mapping = [None] * len(encoded_names)
    transformers = {
        name: (transformer, features)
        for name, transformer, features in columns.transformers_
    }
    for group in ("numeric", "binary"):
        _, features = transformers[group]
        positions = columns.output_indices_[group]
        if positions.stop - positions.start != len(features):
            raise ValueError(f"Unexpected {group} encoded width")
        mapping[positions] = list(features)
    categorical, features = transformers["categorical"]
    encoder = categorical.named_steps["encoder"]
    if encoder.min_frequency is not None or encoder.max_categories is not None:
        raise ValueError("Infrequent category encoding is unsupported")
    start = columns.output_indices_["categorical"].start
    end = columns.output_indices_["categorical"].stop
    drop_indices = encoder.drop_idx_
    for position, feature in enumerate(features):
        dropped = drop_indices is not None and drop_indices[position] is not None
        width = len(encoder.categories_[position]) - int(dropped)
        mapping[start : start + width] = [feature] * width
        start += width
    if start != end or any(name is None for name in mapping):
        raise ValueError("Every transformed feature must map exactly once")
    if set(mapping) != set(expected):
        raise ValueError("Transformed and original predictor sets differ")
    return tuple(mapping)


def aggregate_contributions(
    values: np.ndarray, mapping: tuple[str, ...], names: tuple[str, ...]
) -> dict[str, float]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) != len(mapping) or not np.isfinite(vector).all():
        raise ValueError("Invalid transformed SHAP contributions")
    result = {name: 0.0 for name in names}
    for original, contribution in zip(mapping, vector, strict=True):
        result[original] += float(contribution)
    if not np.isclose(sum(result.values()), float(vector.sum()), atol=1e-8):
        raise ValueError("Original-feature aggregation changed total contribution")
    return result
