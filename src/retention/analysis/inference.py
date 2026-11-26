"""Reusable statistical inference for the development customer population."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from retention.analysis.association import cramers_v


def _finite_sample(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
    """Return a finite one-dimensional sample with at least two observations."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size < 2 or not np.isfinite(sample).all():
        raise ValueError(f"{name} must be a finite one-dimensional sample with n >= 2.")
    return sample


def wilson_interval(
    successes: int, total: int, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion."""
    if not isinstance(successes, (int, np.integer)) or not isinstance(
        total, (int, np.integer)
    ):
        raise TypeError("successes and total must be integers.")
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Require 0 <= successes <= total and total > 0.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")
    proportion = successes / total
    z_value = stats.norm.ppf((1 + confidence_level) / 2)
    denominator = 1 + z_value**2 / total
    center = (proportion + z_value**2 / (2 * total)) / denominator
    margin = (
        z_value
        * np.sqrt(proportion * (1 - proportion) / total + z_value**2 / (4 * total**2))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, float(center - margin))
    upper = 1.0 if successes == total else min(1.0, float(center + margin))
    return lower, upper


def welch_mean_test(
    group_one: pd.Series | np.ndarray,
    group_two: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    """Return Welch inference and Cohen's d for group one minus group two."""
    first = _finite_sample(group_one, "group_one")
    second = _finite_sample(group_two, "group_two")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")
    n_one, n_two = len(first), len(second)
    mean_one, mean_two = first.mean(), second.mean()
    variance_one, variance_two = first.var(ddof=1), second.var(ddof=1)
    variance_term = variance_one / n_one + variance_two / n_two
    if not np.isfinite(variance_term) or variance_term <= 0:
        raise ValueError("Welch inference requires positive finite sampling variance.")
    standard_error = np.sqrt(variance_term)
    degrees_freedom = variance_term**2 / (
        (variance_one / n_one) ** 2 / (n_one - 1)
        + (variance_two / n_two) ** 2 / (n_two - 1)
    )
    difference = mean_one - mean_two
    statistic = difference / standard_error
    p_value = 2 * stats.t.sf(abs(statistic), degrees_freedom)
    critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    pooled_variance = ((n_one - 1) * variance_one + (n_two - 1) * variance_two) / (
        n_one + n_two - 2
    )
    if pooled_variance <= 0:
        raise ValueError("Cohen's d requires positive pooled variance.")
    return {
        "group_one_n": n_one,
        "group_two_n": n_two,
        "group_one_mean": float(mean_one),
        "group_two_mean": float(mean_two),
        "mean_difference": float(difference),
        "standard_error": float(standard_error),
        "degrees_of_freedom": float(degrees_freedom),
        "test_statistic": float(statistic),
        "raw_p_value": float(p_value),
        "ci_lower": float(difference - critical * standard_error),
        "ci_upper": float(difference + critical * standard_error),
        "cohens_d": float(difference / np.sqrt(pooled_variance)),
    }


def mann_whitney_test(
    group_one: pd.Series | np.ndarray, group_two: pd.Series | np.ndarray
) -> dict[str, float | bool | int]:
    """Return tie-aware asymptotic Mann-Whitney inference and Cliff's delta."""
    first = _finite_sample(group_one, "group_one")
    second = _finite_sample(group_two, "group_two")
    result = stats.mannwhitneyu(
        first,
        second,
        alternative="two-sided",
        method="asymptotic",
        use_continuity=True,
    )
    superiority = result.statistic / (len(first) * len(second))
    return {
        "group_one_n": len(first),
        "group_two_n": len(second),
        "test_statistic": float(result.statistic),
        "raw_p_value": float(result.pvalue),
        "probability_of_superiority": float(superiority),
        "cliffs_delta": float(2 * superiority - 1),
        "ties_present": bool(
            np.unique(np.concatenate([first, second])).size < len(first) + len(second)
        ),
    }


def expected_frequency_diagnostics(expected: np.ndarray) -> dict[str, Any]:
    """Evaluate conventional Pearson chi-square expected-frequency guidance."""
    frequencies = np.asarray(expected, dtype=float)
    if (
        frequencies.ndim != 2
        or not frequencies.size
        or not np.isfinite(frequencies).all()
    ):
        raise ValueError("expected must be a finite, non-empty two-dimensional array.")
    below_one = int((frequencies < 1).sum())
    below_five = int((frequencies < 5).sum())
    return {
        "minimum_expected_frequency": float(frequencies.min()),
        "cells_expected_below_1": below_one,
        "cells_expected_below_5": below_five,
        "proportion_expected_below_5": float(below_five / frequencies.size),
        "expected_frequency_assumption_met": bool(
            below_one == 0 and below_five / frequencies.size <= 0.20
        ),
    }


def categorical_independence_test(
    predictor: pd.Series, outcome: pd.Series
) -> dict[str, Any]:
    """Return Pearson chi-square inference and Fisher fallback when justified."""
    if len(predictor) != len(outcome) or len(predictor) == 0:
        raise ValueError("predictor and outcome must have equal positive length.")
    if predictor.isna().any() or outcome.isna().any():
        raise ValueError("categorical inference does not accept missing values.")
    table = pd.crosstab(predictor, outcome, dropna=False)
    if min(table.shape) < 2:
        raise ValueError("Both variables must contain at least two observed levels.")
    statistic, p_value, degrees_freedom, expected = stats.chi2_contingency(
        table.to_numpy(), correction=False
    )
    diagnostics = expected_frequency_diagnostics(expected)
    fisher_needed = (
        table.shape == (2, 2) and not diagnostics["expected_frequency_assumption_met"]
    )
    fisher_odds_ratio: float | None = None
    fisher_p_value: float | None = None
    if fisher_needed:
        fisher = stats.fisher_exact(table.to_numpy(), alternative="two-sided")
        fisher_odds_ratio = float(fisher.statistic)
        fisher_p_value = float(fisher.pvalue)
    return {
        "sample_size": int(table.to_numpy().sum()),
        "table_rows": table.shape[0],
        "table_columns": table.shape[1],
        "test_statistic": float(statistic),
        "degrees_of_freedom": int(degrees_freedom),
        "raw_p_value": float(p_value),
        "bias_corrected_cramers_v": cramers_v(predictor, outcome, bias_corrected=True),
        **diagnostics,
        "fisher_exact_needed": fisher_needed,
        "fisher_odds_ratio": fisher_odds_ratio,
        "fisher_p_value": fisher_p_value,
    }
