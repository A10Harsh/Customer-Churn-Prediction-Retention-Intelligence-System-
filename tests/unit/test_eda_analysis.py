"""Unit tests for reusable automated EDA computations."""

import json

import numpy as np
import pandas as pd
import pytest

from retention.analysis.association import (
    correlation_ratio,
    cramers_v,
    numerical_correlations,
    point_biserial_with_target,
)
from retention.analysis.descriptive import (
    categorical_frequencies,
    numerical_statistics,
    rare_categories,
    target_summary,
    variability_screen,
)
from retention.analysis.outliers import (
    iqr_flags,
    modified_z_scores,
    percentile_flags,
    standard_z_scores,
)
from retention.analysis.reporting import json_safe
from retention.analysis.segmentation import (
    pareto_concentration,
    prepare_segments,
    segment_summary,
    verify_reconciliation,
)


def test_numerical_statistics_uses_sample_conventions() -> None:
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    result = numerical_statistics(data, ["x"]).loc["x"]
    assert result["mean"] == pytest.approx(2.5)
    assert result["median"] == pytest.approx(2.5)
    assert result["variance"] == pytest.approx(5 / 3)
    assert result["iqr"] == pytest.approx(1.5)
    assert result["skewness"] == pytest.approx(0.0)


def test_target_summary_and_majority_baseline() -> None:
    result = target_summary(pd.DataFrame({"Churn": ["No"] * 3 + ["Yes"]}), "Churn")
    assert result["positive_rate"] == 0.25
    assert result["imbalance_ratio"] == 3.0
    assert result["majority_baseline"]["accuracy"] == 0.75
    assert result["majority_baseline"]["positive_recall"] == 0.0
    assert result["majority_baseline"]["balanced_accuracy"] == 0.5


def test_target_summary_rejects_nonbinary_target() -> None:
    with pytest.raises(ValueError):
        target_summary(pd.DataFrame({"Churn": ["No", "No"]}), "Churn")


def test_rare_categories_use_strict_threshold() -> None:
    frame = pd.DataFrame(
        {
            "feature": ["x", "x"],
            "category": ["a", "b"],
            "count": [99, 1],
            "percentage": [99.0, 1.0],
        }
    )
    assert rare_categories(frame, 0.01).empty
    assert rare_categories(frame, 0.02)["category"].tolist() == ["b"]


def test_categorical_frequency_percentages_reconcile() -> None:
    result = categorical_frequencies(pd.DataFrame({"x": ["a", "a", "b"]}), ["x"])
    assert result["count"].sum() == 3
    assert result["percentage"].sum() == pytest.approx(100)


def test_constant_and_near_constant_screen() -> None:
    data = pd.DataFrame({"constant": [1] * 10, "dominant": [1] * 9 + [2]})
    result = variability_screen(data, list(data), 0.90)
    assert result.loc["constant", "constant"]
    assert result.loc["dominant", "near_constant"]


def test_correlations_and_point_biserial() -> None:
    data = pd.DataFrame(
        {"x": [0, 1, 2, 3], "y": [0, 2, 4, 6], "Churn": ["No", "No", "Yes", "Yes"]}
    )
    result = numerical_correlations(data, ["x", "y"])
    assert result["pearson"].loc["x", "y"] == pytest.approx(1.0)
    assert result["kendall_tau_b"].loc["x", "y"] == pytest.approx(1.0)
    assert point_biserial_with_target(data, ["x"], "Churn").iloc[0, 0] > 0


def test_constant_associations_are_safe() -> None:
    assert cramers_v(pd.Series(["x"] * 4), pd.Series(["a", "b"] * 2)) == 0.0
    assert correlation_ratio(pd.Series(["a", "b"]), pd.Series([5.0, 5.0])) == 0.0


def test_segmentation_boundaries_and_lift() -> None:
    data = _segment_frame([0, 6, 7, 12, 13, 24, 25, 48, 49, 72])
    prepared = prepare_segments(data, "Churn", [-0.001, 6, 12, 24, 48, 72])
    assert prepared["tenure_segment"].astype(str).tolist() == [
        "0–6: Onboarding",
        "0–6: Onboarding",
        "7–12: First year",
        "7–12: First year",
        "13–24: Developing",
        "13–24: Developing",
        "25–48: Established",
        "25–48: Established",
        "49–72: Mature",
        "49–72: Mature",
    ]
    summary = segment_summary(prepared, "tenure_segment")
    assert summary.loc["0–6: Onboarding", "churn_rate_lift"] == pytest.approx(1.0)


def test_segment_exposure_reconciliation() -> None:
    prepared = prepare_segments(
        _segment_frame(list(range(10))), "Churn", [-0.001, 6, 12, 24, 48, 72]
    )
    summary = segment_summary(prepared, "tenure_segment")
    verify_reconciliation(prepared, summary)
    assert summary["total_monthly_charge_exposure"].sum() == pytest.approx(
        prepared["MonthlyCharges"].sum()
    )


def test_sparse_groups_are_explicit_and_pareto_is_monotonic() -> None:
    prepared = prepare_segments(
        _segment_frame([0, 72]), "Churn", [-0.001, 6, 12, 24, 48, 72]
    )
    table = segment_summary(prepared, ["Contract", "tenure_segment"]).reset_index()
    pareto = pareto_concentration(table, "churned_count")
    assert len(table) == 2
    assert pareto["cumulative_contribution"].is_monotonic_increasing


def test_iqr_and_percentile_flags_are_strict() -> None:
    data = pd.DataFrame({"x": [0, 1, 2, 3, 100]})
    summary, flags = iqr_flags(data, ["x"], 1.5)
    assert summary.loc["x", "flagged_count"] == 1
    percentile, percentile_flag = percentile_flags(data, ["x"], 0.01, 0.99)
    assert percentile.loc["x", "flagged_count"] == 2
    assert percentile_flag["x"].sum() == 2


def test_zero_variance_and_zero_mad_are_safe() -> None:
    values = pd.Series([7.0, 7.0, 7.0])
    assert standard_z_scores(values).eq(0).all()
    modified, median, mad = modified_z_scores(values)
    assert modified.eq(0).all() and median == 7 and mad == 0


def test_json_safe_replaces_nonfinite_scalars() -> None:
    result = json_safe({"nan": np.nan, "inf": np.inf, "value": np.int64(2)})
    assert result == {"nan": None, "inf": None, "value": 2}
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def _segment_frame(tenures: list[int]) -> pd.DataFrame:
    size = len(tenures)
    yes_no = ["Yes" if i % 2 else "No" for i in range(size)]
    return pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(size)],
            "tenure": tenures,
            "MonthlyCharges": np.arange(size, dtype=float) + 20,
            "TotalCharges": np.arange(size, dtype=float) * 100,
            "Churn": yes_no,
            "Contract": ["Month-to-month"] * size,
            "PhoneService": ["Yes"] * size,
            "MultipleLines": ["No"] * size,
            "InternetService": ["DSL"] * size,
            "OnlineSecurity": yes_no,
            "OnlineBackup": ["No"] * size,
            "DeviceProtection": ["No"] * size,
            "TechSupport": ["No"] * size,
            "StreamingTV": ["No"] * size,
            "StreamingMovies": ["No"] * size,
        }
    )
