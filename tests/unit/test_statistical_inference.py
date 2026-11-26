"""Focused tests for reusable statistical inference."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from retention.analysis.inference import (
    categorical_independence_test,
    expected_frequency_diagnostics,
    mann_whitney_test,
    welch_mean_test,
    wilson_interval,
)


def test_welch_and_effect_orientation_match_scipy() -> None:
    churned = np.array([1.0, 2.0, 3.0, 5.0])
    retained = np.array([5.0, 6.0, 8.0, 9.0, 11.0])
    result = welch_mean_test(churned, retained)
    expected = stats.ttest_ind(churned, retained, equal_var=False)
    assert result["test_statistic"] == pytest.approx(expected.statistic)
    assert result["raw_p_value"] == pytest.approx(expected.pvalue)
    assert result["mean_difference"] < 0
    assert result["cohens_d"] < 0
    assert result["ci_lower"] < result["mean_difference"] < result["ci_upper"]


def test_mann_whitney_and_cliffs_delta_orientation() -> None:
    result = mann_whitney_test([5, 6, 7], [1, 2, 3])
    assert result["test_statistic"] == 9
    assert result["probability_of_superiority"] == 1
    assert result["cliffs_delta"] == 1
    assert not result["ties_present"]


@pytest.mark.parametrize("successes", [0, 10])
def test_wilson_interval_respects_probability_boundaries(successes: int) -> None:
    lower, upper = wilson_interval(successes, 10)
    assert 0 <= lower <= upper <= 1
    assert lower == 0 if successes == 0 else upper == 1


def test_wilson_interval_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(2, 1)
    with pytest.raises(ValueError):
        wilson_interval(1, 2, confidence_level=1.0)
    with pytest.raises(TypeError):
        wilson_interval(1.0, 2)  # type: ignore[arg-type]


def test_chi_square_expected_frequencies_and_fisher_decision() -> None:
    predictor = pd.Series(["A"] * 2 + ["B"] * 8)
    outcome = pd.Series(["Yes", "No"] + ["No"] * 8)
    result = categorical_independence_test(predictor, outcome)
    assert not result["expected_frequency_assumption_met"]
    assert result["fisher_exact_needed"]
    assert result["fisher_p_value"] is not None
    assert 0 <= result["bias_corrected_cramers_v"] <= 1


def test_expected_frequency_validation_rejects_invalid_array() -> None:
    with pytest.raises(ValueError):
        expected_frequency_diagnostics(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        expected_frequency_diagnostics(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError):
        categorical_independence_test(pd.Series(["A", "A"]), pd.Series(["No", "No"]))


def test_inference_rejects_missing_mismatched_and_short_inputs() -> None:
    with pytest.raises(ValueError):
        mann_whitney_test([1.0], [2.0, 3.0])
    with pytest.raises(ValueError):
        welch_mean_test([1.0, 2.0], [3.0, 4.0], confidence_level=0.0)
    with pytest.raises(ValueError):
        categorical_independence_test(pd.Series(["A"]), pd.Series(["No", "Yes"]))
    with pytest.raises(ValueError):
        categorical_independence_test(pd.Series(["A", None]), pd.Series(["No", "Yes"]))


def test_welch_rejects_degenerate_variance() -> None:
    with pytest.raises(ValueError):
        welch_mean_test([1.0, 1.0], [1.0, 1.0])
