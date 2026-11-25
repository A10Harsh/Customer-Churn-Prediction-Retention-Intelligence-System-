"""Shared test fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def valid_customer_df() -> pd.DataFrame:
    """Return a valid synthetic customer dataset."""

    return pd.DataFrame(
        [
            {
                "customerID": "TEST-001",
                "gender": "Male",
                "SeniorCitizen": "0",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": "12",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": "55.50",
                "TotalCharges": "666.00",
                "Churn": "No",
            }
        ],
        dtype="string",
    )
