"""Focused tests for deterministic candidate feature engineering."""

import numpy as np
import pandas as pd
import pytest

from retention.features.engineering import TelcoFeatureEngineer


def customer(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "customerID": "A",
        "tenure": 6,
        "TotalCharges": 120.0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaymentMethod": "Credit card (automatic)",
        "Churn": "Yes",
    }
    row.update(changes)
    return row


def test_calculations_and_exclusions() -> None:
    data = pd.DataFrame([customer()])
    original = data.copy(deep=True)
    transformer = TelcoFeatureEngineer().fit(data)
    result = transformer.transform(data)

    assert result.loc[0, "tenure_group"] == "0–6: Onboarding"
    assert result.loc[0, "total_services"] == 4
    assert result.loc[0, "support_security_count"] == 2
    assert result.loc[0, "is_auto_payment"] == 1
    assert result.loc[0, "is_month_to_month"] == 1
    assert result.loc[0, "avg_monthly_spend"] == 20
    assert result.loc[0, "OnlineSecurity_adopted"] == 1
    assert "customerID" not in result and "Churn" not in result
    assert "OnlineSecurity" not in result and "MultipleLines" not in result
    assert result.loc[0, "PhoneService"] == "Yes"
    assert result.loc[0, "InternetService"] == "DSL"
    assert list(result.columns) == list(transformer.get_feature_names_out())
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(
    ("tenure", "expected"),
    [
        (0, "0–6: Onboarding"),
        (6, "0–6: Onboarding"),
        (7, "7–12: First year"),
        (12, "7–12: First year"),
        (13, "13–24: Developing"),
        (24, "13–24: Developing"),
        (25, "25–48: Established"),
        (48, "25–48: Established"),
        (49, "49–72: Mature"),
        (72, "49–72: Mature"),
    ],
)
def test_lifecycle_boundaries(tenure: int, expected: str) -> None:
    result = TelcoFeatureEngineer().fit_transform(
        pd.DataFrame([customer(tenure=tenure)])
    )
    assert result.loc[0, "tenure_group"] == expected


def test_zero_and_missing_numeric_values() -> None:
    data = pd.DataFrame(
        [
            customer(tenure=0, TotalCharges=0.0),
            customer(tenure=np.nan, TotalCharges=np.nan),
        ]
    )
    result = TelcoFeatureEngineer().fit_transform(data)
    assert result["avg_monthly_spend"].isna().all()
    assert pd.isna(result.loc[1, "tenure_group"])


def test_service_count_and_structural_states() -> None:
    no_services = customer(
        PhoneService="No",
        MultipleLines="No phone service",
        InternetService="No",
        **{
            column: "No internet service"
            for column in (
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            )
        },
    )
    available_no_adoption = customer(
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="DSL",
        OnlineSecurity="No",
        TechSupport="No",
    )
    all_services = customer(
        MultipleLines="Yes",
        InternetService="Fiber optic",
        **{
            column: "Yes"
            for column in (
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            )
        },
    )
    result = TelcoFeatureEngineer().fit_transform(
        pd.DataFrame([no_services, available_no_adoption, all_services])
    )
    assert result["total_services"].tolist() == [0, 2, 9]
    assert result["support_security_count"].tolist() == [0, 0, 2]
    assert result["multiple_lines_adopted"].tolist() == [0, 0, 1]
    assert result["PhoneService"].tolist() == ["No", "Yes", "Yes"]
    assert result["InternetService"].tolist() == ["No", "DSL", "Fiber optic"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"OnlineSecurity": "Unknown"}, "OnlineSecurity"),
        ({"OnlineSecurity": None}, "OnlineSecurity"),
        ({"MultipleLines": "No phone service"}, "inconsistent"),
        ({"InternetService": "No"}, "inconsistent"),
        ({"tenure": -1}, "tenure"),
        ({"TotalCharges": -1}, "TotalCharges"),
        ({"tenure": 73}, "lifecycle"),
    ],
)
def test_invalid_input_is_explicit(changes: dict[str, object], message: str) -> None:
    data = pd.DataFrame([customer()])
    transformer = TelcoFeatureEngineer().fit(data)
    data = pd.DataFrame([customer(**changes)])
    with pytest.raises(ValueError, match=message):
        transformer.transform(data)


def test_output_is_consistent_with_or_without_target_and_id() -> None:
    data = pd.DataFrame([customer(), customer(tenure=12, Churn="No")])
    transformer = TelcoFeatureEngineer().fit(data)
    expected = transformer.transform(data)
    actual = transformer.transform(data.drop(columns=["customerID", "Churn"]))
    pd.testing.assert_frame_equal(actual, expected)
    assert transformer.n_features_in_ == 13
    assert len(transformer.get_feature_names_out()) == 19
