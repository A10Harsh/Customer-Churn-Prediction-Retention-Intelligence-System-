"""Pandera schema for the IBM Telco dataset."""

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Check

YES_NO = ["Yes", "No"]

INTERNET_OPTIONS = [
    "Yes",
    "No",
    "No internet service",
]

INTERNET_DEPENDENT_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def categorical_column(
    allowed_values: list[str],
) -> pa.Column:
    """Create a categorical column constraint."""

    return pa.Column(
        pd.StringDtype(),
        checks=Check.isin(allowed_values),
        nullable=False,
    )


def numeric_column() -> pa.Column:
    """Create a finite, non-negative numeric constraint."""

    return pa.Column(
        float,
        checks=[
            Check.ge(0),
            Check(
                lambda series: np.isfinite(series),
                name="finite_values",
            ),
        ],
        nullable=False,
    )


def phone_service_consistency(
    df: pd.DataFrame,
) -> pd.Series:
    """Validate phone service and multiple-line compatibility."""

    return (df["PhoneService"] == "No") == (df["MultipleLines"] == "No phone service")


def internet_service_consistency(
    df: pd.DataFrame,
) -> pd.Series:
    """Validate internet-dependent service compatibility."""

    no_internet = df["InternetService"] == "No"

    valid = pd.Series(
        True,
        index=df.index,
        dtype=bool,
    )

    for column in INTERNET_DEPENDENT_COLUMNS:
        valid &= no_internet == (df[column] == "No internet service")

    return valid


TELCO_SCHEMA = pa.DataFrameSchema(
    columns={
        "customerID": pa.Column(
            pd.StringDtype(),
            checks=Check.str_length(min_value=1),
            unique=True,
            nullable=False,
        ),
        "gender": categorical_column(["Male", "Female"]),
        "SeniorCitizen": pa.Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
        ),
        "Partner": categorical_column(YES_NO),
        "Dependents": categorical_column(YES_NO),
        "tenure": pa.Column(
            int,
            checks=Check.ge(0),
            nullable=False,
        ),
        "PhoneService": categorical_column(YES_NO),
        "MultipleLines": categorical_column(["Yes", "No", "No phone service"]),
        "InternetService": categorical_column(["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": categorical_column(INTERNET_OPTIONS),
        "OnlineBackup": categorical_column(INTERNET_OPTIONS),
        "DeviceProtection": categorical_column(INTERNET_OPTIONS),
        "TechSupport": categorical_column(INTERNET_OPTIONS),
        "StreamingTV": categorical_column(INTERNET_OPTIONS),
        "StreamingMovies": categorical_column(INTERNET_OPTIONS),
        "Contract": categorical_column(["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": categorical_column(YES_NO),
        "PaymentMethod": categorical_column(PAYMENT_METHODS),
        "MonthlyCharges": numeric_column(),
        "TotalCharges": numeric_column(),
        "Churn": categorical_column(YES_NO),
    },
    checks=[
        Check(
            lambda df: not df.empty,
            name="non_empty_dataset",
        ),
        Check(
            phone_service_consistency,
            name="phone_service_consistency",
        ),
        Check(
            internet_service_consistency,
            name="internet_service_consistency",
        ),
    ],
    strict=True,
    coerce=False,
    unique_column_names=True,
    name="IBM_Telco_Customer_Churn",
)
