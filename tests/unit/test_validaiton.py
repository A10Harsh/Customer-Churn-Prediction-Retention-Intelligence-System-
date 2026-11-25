"""Unit tests for dataset validation."""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from retention.data.validation import (
    DataValidationError,
    prepare_data,
    validate_data,
)


def test_valid_dataset(
    valid_customer_df: pd.DataFrame,
) -> None:
    cleaned, _ = prepare_data(valid_customer_df)
    validated = validate_data(cleaned)

    assert validated.shape == (1, 21)


def test_numeric_conversion(
    valid_customer_df: pd.DataFrame,
) -> None:
    cleaned, _ = prepare_data(valid_customer_df)

    assert cleaned["tenure"].dtype == "int64"
    assert cleaned["TotalCharges"].dtype == "float64"


def test_zero_tenure_missing_charges(
    valid_customer_df: pd.DataFrame,
) -> None:
    valid_customer_df.loc[0, "tenure"] = "0"
    valid_customer_df.loc[0, "TotalCharges"] = pd.NA

    cleaned, corrections = prepare_data(valid_customer_df)

    assert cleaned.loc[0, "TotalCharges"] == 0.0

    assert corrections["zero_tenure_total_charges_filled"] == 1

    validate_data(cleaned)


def test_unexpected_missing_charges(
    valid_customer_df: pd.DataFrame,
) -> None:
    valid_customer_df.loc[0, "TotalCharges"] = pd.NA

    cleaned, _ = prepare_data(valid_customer_df)

    with pytest.raises(SchemaErrors):
        validate_data(cleaned)


def test_invalid_numeric_value(
    valid_customer_df: pd.DataFrame,
) -> None:
    valid_customer_df.loc[0, "TotalCharges"] = "abc"

    with pytest.raises(DataValidationError):
        prepare_data(valid_customer_df)


def test_fractional_tenure(
    valid_customer_df: pd.DataFrame,
) -> None:
    valid_customer_df.loc[0, "tenure"] = "2.5"

    with pytest.raises(DataValidationError):
        prepare_data(valid_customer_df)


def test_missing_column(
    valid_customer_df: pd.DataFrame,
) -> None:
    df = valid_customer_df.drop(columns=["Churn"])

    with pytest.raises(DataValidationError):
        prepare_data(df)


def test_unexpected_column(
    valid_customer_df: pd.DataFrame,
) -> None:
    valid_customer_df["unknown"] = "test"

    with pytest.raises(DataValidationError):
        prepare_data(valid_customer_df)


def test_duplicate_customer_id(
    valid_customer_df: pd.DataFrame,
) -> None:
    df = pd.concat(
        [valid_customer_df, valid_customer_df],
        ignore_index=True,
    )

    cleaned, _ = prepare_data(df)

    with pytest.raises(SchemaErrors):
        validate_data(cleaned)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Churn", "Unknown"),
        ("SeniorCitizen", "2"),
        ("gender", "Invalid"),
        ("Contract", "Invalid"),
        ("MonthlyCharges", "-10"),
        ("tenure", "-5"),
        ("MonthlyCharges", "inf"),
    ],
)
def test_invalid_column_values(
    valid_customer_df: pd.DataFrame,
    column: str,
    value: str,
) -> None:
    valid_customer_df.loc[0, column] = value

    cleaned, _ = prepare_data(valid_customer_df)

    with pytest.raises(SchemaErrors):
        validate_data(cleaned)


def test_phone_service_consistency(
    valid_customer_df: pd.DataFrame,
) -> None:
    valid_customer_df.loc[0, "PhoneService"] = "No"
    valid_customer_df.loc[0, "MultipleLines"] = "Yes"

    cleaned, _ = prepare_data(valid_customer_df)

    with pytest.raises(SchemaErrors):
        validate_data(cleaned)


def test_internet_service_consistency(
    valid_customer_df: pd.DataFrame,
) -> None:
    valid_customer_df.loc[0, "InternetService"] = "No"
    valid_customer_df.loc[0, "StreamingTV"] = "Yes"

    cleaned, _ = prepare_data(valid_customer_df)

    with pytest.raises(SchemaErrors):
        validate_data(cleaned)


def test_disabled_correction(
    valid_customer_df: pd.DataFrame,
) -> None:
    valid_customer_df.loc[0, "tenure"] = "0"
    valid_customer_df.loc[0, "TotalCharges"] = pd.NA

    cleaned, corrections = prepare_data(
        valid_customer_df,
        fill_zero_tenure_total_charges=False,
    )

    assert corrections["zero_tenure_total_charges_filled"] == 0

    with pytest.raises(SchemaErrors):
        validate_data(cleaned)
