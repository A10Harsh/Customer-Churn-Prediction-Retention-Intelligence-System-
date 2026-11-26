"""VIF and design-matrix diagnostics without persistent preprocessing."""

from typing import Any

import numpy as np
import pandas as pd


def calculate_vif(design: pd.DataFrame, exact_tolerance: float = 1e-12) -> pd.DataFrame:
    """Calculate auxiliary R-squared, VIF, and tolerance with an intercept."""
    if design.empty or design.shape[1] < 2:
        raise ValueError("VIF requires at least two predictor columns and one row.")
    values = design.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("VIF design must contain only finite values.")
    if exact_tolerance <= 0:
        raise ValueError("exact_tolerance must be positive.")
    rows = []
    for index, predictor in enumerate(design.columns):
        outcome = values[:, index]
        total_sum_squares = np.sum((outcome - outcome.mean()) ** 2)
        if total_sum_squares <= np.finfo(float).eps * max(1.0, np.sum(outcome**2)):
            rows.append(
                {
                    "predictor": predictor,
                    "auxiliary_r_squared": np.nan,
                    "vif": np.inf,
                    "tolerance": 0.0,
                    "dependency_type": "constant",
                }
            )
            continue
        others = np.delete(values, index, axis=1)
        auxiliary = np.column_stack([np.ones(len(outcome)), others])
        coefficients = np.linalg.lstsq(auxiliary, outcome, rcond=None)[0]
        residual_sum_squares = np.sum((outcome - auxiliary @ coefficients) ** 2)
        residual_ratio = max(0.0, residual_sum_squares / total_sum_squares)
        if residual_ratio <= exact_tolerance:
            r_squared, vif, tolerance = 1.0, np.inf, 0.0
            dependency = "exact_linear_dependency"
        else:
            r_squared = float(np.clip(1 - residual_ratio, 0.0, 1.0))
            tolerance = 1 - r_squared
            vif = 1 / tolerance
            dependency = "finite"
        rows.append(
            {
                "predictor": predictor,
                "auxiliary_r_squared": r_squared,
                "vif": vif,
                "tolerance": tolerance,
                "dependency_type": dependency,
            }
        )
    return pd.DataFrame(rows)


def reference_coded_design(
    data: pd.DataFrame, numerical: list[str], categorical: list[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a temporary reference-coded predictor matrix and its references."""
    requested = numerical + categorical
    missing = sorted(set(requested) - set(data.columns))
    if missing:
        raise ValueError(f"Design columns are missing: {missing}")
    if len(requested) != len(set(requested)):
        raise ValueError("Numerical and categorical predictor lists must be disjoint.")
    categories = data[categorical].astype("category")
    references = {
        column: str(categories[column].cat.categories[0]) for column in categorical
    }
    encoded = pd.get_dummies(categories, drop_first=True, dtype=float)
    design = pd.concat([data[numerical].astype(float), encoded], axis=1)
    if not np.isfinite(design.to_numpy(dtype=float)).all():
        raise ValueError("Encoded design contains missing or non-finite values.")
    return design, references


def matrix_diagnostics(design: pd.DataFrame) -> dict[str, Any]:
    """Return standardized singular-value, rank, and condition diagnostics."""
    if design.empty:
        raise ValueError("Design matrix cannot be empty.")
    values = design.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Design matrix must contain only finite values.")
    standard_deviations = values.std(axis=0, ddof=0)
    constant_columns = design.columns[
        standard_deviations <= np.finfo(float).eps
    ].tolist()
    nonconstant = standard_deviations > np.finfo(float).eps
    standardized = (
        (values[:, nonconstant] - values[:, nonconstant].mean(axis=0))
        / standard_deviations[nonconstant]
        if nonconstant.any()
        else np.empty((len(design), 0))
    )
    with_intercept = np.column_stack([np.ones(len(design)), standardized])
    singular_values = np.linalg.svd(with_intercept, compute_uv=False)
    rank = int(np.linalg.matrix_rank(with_intercept))
    column_count = with_intercept.shape[1]
    condition_number = (
        np.inf
        if rank < column_count
        else float(singular_values[0] / singular_values[-1])
    )
    return {
        "rows": len(design),
        "predictor_columns": design.shape[1],
        "diagnostic_columns_with_intercept": column_count,
        "rank": rank,
        "rank_deficiency": column_count - rank,
        "condition_number": condition_number,
        "largest_singular_value": float(singular_values[0]),
        "smallest_singular_value": float(singular_values[-1]),
        "constant_columns": constant_columns,
    }
