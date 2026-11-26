"""Candidate Telco features for later, leakage-safe model comparisons.

The output retains the original predictors except the seven dependent service
statuses. PhoneService and InternetService retain availability and internet type;
binary adoption fields replace MultipleLines and the six internet add-on statuses.
The unchanged input remains the baseline representation.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from retention.analysis.segmentation import LIFECYCLE_LABELS
from retention.data.schemas import (
    INTERNET_DEPENDENT_COLUMNS,
    PAYMENT_METHODS,
    internet_service_consistency,
    phone_service_consistency,
)
from retention.utils.config import load_config

_SERVICE_COLUMNS = ["MultipleLines", *INTERNET_DEPENDENT_COLUMNS]
_CATEGORIES = {
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{
        column: ["Yes", "No", "No internet service"]
        for column in INTERNET_DEPENDENT_COLUMNS
    },
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": PAYMENT_METHODS,
}
_REQUIRED = ["tenure", "TotalCharges", *_CATEGORIES]
_DERIVED = [
    "tenure_group",
    "total_services",
    "support_security_count",
    "is_auto_payment",
    "is_month_to_month",
    "avg_monthly_spend",
    "multiple_lines_adopted",
    *[f"{column}_adopted" for column in INTERNET_DEPENDENT_COLUMNS],
]


class TelcoFeatureEngineer(TransformerMixin, BaseEstimator):
    """Build deterministic candidate features without using Churn or customerID.

    Missing numeric values propagate to derived numeric fields; zero tenure
    produces a missing average monthly spend. Missing or invalid service,
    contract, and payment categories, inconsistent availability, and negative
    numeric values raise ValueError. Imputation belongs in a later model pipeline.
    """

    def __init__(self, config_path: str = "params.yaml") -> None:
        self.config_path = config_path

    def fit(self, X: pd.DataFrame, y: object = None) -> "TelcoFeatureEngineer":
        """Record predictor schema and the configured lifecycle boundaries."""
        config = load_config(self.config_path)
        self.id_column_ = config["data"]["id_column"]
        self.target_column_ = config["data"]["target_column"]
        self.tenure_boundaries_ = config["eda"]["tenure_boundaries"]
        if len(self.tenure_boundaries_) != len(LIFECYCLE_LABELS) + 1:
            raise ValueError("Lifecycle boundaries do not match lifecycle labels.")
        predictors = self._predictors(X)
        self._validate(predictors)
        self.feature_names_in_ = np.asarray(predictors.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.feature_names_out_ = np.asarray(
            [column for column in predictors if column not in _SERVICE_COLUMNS]
            + _DERIVED,
            dtype=object,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return model-ready candidate columns with stable names and order."""
        check_is_fitted(self, "feature_names_out_")
        predictors = self._predictors(X)
        if list(predictors.columns) != list(self.feature_names_in_):
            raise ValueError("Predictor columns or their order differ from fit input.")
        self._validate(predictors)

        result = predictors.drop(columns=_SERVICE_COLUMNS).copy()
        tenure = pd.to_numeric(predictors["tenure"])
        charges = pd.to_numeric(predictors["TotalCharges"])
        result["tenure_group"] = pd.cut(
            tenure,
            bins=self.tenure_boundaries_,
            labels=LIFECYCLE_LABELS,
            include_lowest=True,
            ordered=True,
        )
        if (tenure.notna() & result["tenure_group"].isna()).any():
            raise ValueError("tenure falls outside configured lifecycle boundaries.")

        # Count phone, multiple lines, internet, and each adopted internet add-on
        # once. Unavailable and available-but-unsubscribed services count zero.
        adopted = {
            column: predictors[column].eq("Yes").astype("int8")
            for column in _SERVICE_COLUMNS
        }
        result["total_services"] = (
            predictors["PhoneService"].eq("Yes").astype("int8")
            + predictors["InternetService"].ne("No").astype("int8")
            + sum(adopted.values())
        )
        result["support_security_count"] = (
            adopted["TechSupport"] + adopted["OnlineSecurity"]
        )
        result["is_auto_payment"] = (
            predictors["PaymentMethod"]
            .isin(["Bank transfer (automatic)", "Credit card (automatic)"])
            .astype("int8")
        )
        result["is_month_to_month"] = (
            predictors["Contract"].eq("Month-to-month").astype("int8")
        )
        result["avg_monthly_spend"] = charges.div(tenure.where(tenure.ne(0)))
        for column in _SERVICE_COLUMNS:
            name = (
                "multiple_lines_adopted"
                if column == "MultipleLines"
                else f"{column}_adopted"
            )
            result[name] = adopted[column]
        return result.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names in transformation order."""
        check_is_fitted(self, "feature_names_out_")
        if input_features is not None and list(input_features) != list(
            self.feature_names_in_
        ):
            raise ValueError("input_features differ from fitted predictor columns.")
        return self.feature_names_out_.copy()

    def _predictors(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TelcoFeatureEngineer requires a pandas DataFrame.")
        if not X.columns.is_unique:
            raise ValueError("Input column names must be unique.")
        return X.drop(columns=[self.id_column_, self.target_column_], errors="ignore")

    @staticmethod
    def _validate(X: pd.DataFrame) -> None:
        missing = set(_REQUIRED) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required predictors: {sorted(missing)}")
        for column, allowed in _CATEGORIES.items():
            if X[column].isna().any() or not X[column].isin(allowed).all():
                raise ValueError(f"Invalid or missing category in {column}.")
        if not phone_service_consistency(X).all():
            raise ValueError("PhoneService and MultipleLines are inconsistent.")
        if not internet_service_consistency(X).all():
            raise ValueError("InternetService and add-on statuses are inconsistent.")
        for column in ("tenure", "TotalCharges"):
            values = pd.to_numeric(X[column], errors="coerce")
            if (X[column].notna() & values.isna()).any() or (values.dropna() < 0).any():
                raise ValueError(f"{column} must be non-negative numeric or missing.")
            if not np.isfinite(values.dropna()).all():
                raise ValueError(f"{column} must contain finite numeric values.")
