"""Unit tests for family-level multiple-testing corrections."""

import numpy as np
import pandas as pd
import pytest

from retention.analysis.multiplicity import (
    SMALLEST_POSITIVE_FLOAT,
    adjust_hypothesis_families,
    adjustment_ready,
    benjamini_hochberg,
    holm,
)


def test_bh_and_holm_match_analytic_example() -> None:
    values = [0.01, 0.04, 0.03, 0.002]
    assert np.allclose(benjamini_hochberg(values), [0.02, 0.04, 0.04, 0.008])
    assert np.allclose(holm(values), [0.03, 0.06, 0.06, 0.008])


def test_underflow_is_bounded_not_treated_as_literal_zero() -> None:
    prepared = adjustment_ready([0.0, 0.5])
    assert prepared[0] == SMALLEST_POSITIVE_FLOAT
    assert benjamini_hochberg([0.0, 0.5])[0] > 0
    assert holm([0.0, 0.5])[0] > 0


def test_family_adjustment_is_separate() -> None:
    tests = pd.DataFrame(
        {
            "hypothesis_id": ["A", "B", "C"],
            "hypothesis_family": ["one", "one", "two"],
            "raw_p_value": [0.04, 0.06, 0.04],
        }
    )
    result = adjust_hypothesis_families(tests, 0.05, 0.05)
    assert result.loc[result["hypothesis_id"].eq("A"), "bh_reject"].item() is False
    assert result.loc[result["hypothesis_id"].eq("C"), "bh_reject"].item() is True


@pytest.mark.parametrize("values", [[], [-0.1], [1.1], [np.nan]])
def test_invalid_p_values_are_rejected(values: list[float]) -> None:
    with pytest.raises(ValueError):
        adjustment_ready(values)
