"""Unfitted, cloneable preprocessing for baseline and engineered Telco features.

Use split_features_target before cross-validation, then place the returned
preprocessor inside an estimator Pipeline. Fit that Pipeline separately in each
training fold. No learned values are computed by these factories.

Missing numeric values use the training fold's median; missing categorical and
binary values use its most frequent value. Unseen categorical values encode as
all zeros. With reference coding this is also the dropped category's encoding,
so interpretation of an unseen value is limited. The engineered path rejects
unknown structural service, contract, or payment categories in engineering.py.
"""

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retention.data.schemas import INTERNET_DEPENDENT_COLUMNS
from retention.features.engineering import (
    EnhancedTelcoFeatureEngineer,
    TelcoFeatureEngineer,
)
from retention.utils.config import load_config

_REPLACED_SERVICES = ["MultipleLines", *INTERNET_DEPENDENT_COLUMNS]
_ENGINEERED_NUMERIC = [
    "total_services",
    "support_security_count",
    "avg_monthly_spend",
]
_ENGINEERED_BINARY = [
    "is_auto_payment",
    "is_month_to_month",
    "multiple_lines_adopted",
    *[f"{column}_adopted" for column in INTERNET_DEPENDENT_COLUMNS],
]


def split_features_target(
    data: pd.DataFrame, config_path: str = "params.yaml"
) -> tuple[pd.DataFrame, pd.Series]:
    """Copy predictors without ID/target and map Churn No/Yes to integer 0/1."""
    config = load_config(config_path)
    target = config["data"]["target_column"]
    identifier = config["data"]["id_column"]
    if target not in data:
        raise ValueError(f"Missing target column: {target}")
    if data[target].isna().any() or not data[target].isin(["No", "Yes"]).all():
        raise ValueError(
            f"{target} must contain only No or Yes, without missing values."
        )
    features = data.drop(columns=[target, identifier], errors="ignore").copy()
    labels = data[target].map({"No": 0, "Yes": 1}).astype("int8").copy()
    return features, labels


def make_preprocessor(
    feature_set: str = "baseline",
    model_family: str = "linear",
    selected_features: Sequence[str] | None = None,
    config_path: str = "params.yaml",
) -> Pipeline:
    """Create an unfitted preprocessing Pipeline for later estimator pipelines.

    feature_set is baseline or engineered. model_family is linear, neural, or
    tree. selected_features explicitly limits encoded input columns; omission
    keeps every predictor. A subset must be chosen inside model evaluation CV.
    """
    if feature_set not in {"baseline", "engineered", "enhanced"}:
        raise ValueError("feature_set must be baseline, engineered, or enhanced.")
    if model_family not in {"linear", "neural", "tree"}:
        raise ValueError("model_family must be linear, neural, or tree.")

    config = load_config(config_path)
    numerical = list(config["eda"]["numerical_features"])
    categorical = list(config["eda"]["categorical_features"])
    binary = ["SeniorCitizen"]
    categorical.remove("SeniorCitizen")
    if feature_set in {"engineered", "enhanced"}:
        numerical += _ENGINEERED_NUMERIC
        categorical = [c for c in categorical if c not in _REPLACED_SERVICES]
        categorical += ["tenure_group"]
        binary += _ENGINEERED_BINARY
    if feature_set == "enhanced":
        numerical += ["monthly_charge_per_service"]
        categorical += [
            "contract_tenure",
            "payment_tenure",
            "internet_charge_band",
            "support_security_status",
        ]
        binary += ["new_customer"]

    available = numerical + binary + categorical
    if len(available) != len(set(available)):
        raise ValueError("Configured feature names must be unique.")
    if selected_features is not None:
        chosen = list(selected_features)
        if not chosen or len(chosen) != len(set(chosen)):
            raise ValueError("selected_features must contain unique feature names.")
        unknown = set(chosen) - set(available)
        if unknown:
            raise ValueError(f"Unknown selected features: {sorted(unknown)}")
        numerical = [c for c in numerical if c in chosen]
        binary = [c for c in binary if c in chosen]
        categorical = [c for c in categorical if c in chosen]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if model_family != "tree":
        numeric_steps.append(("scaler", StandardScaler()))
    category_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    missing_values=pd.NA,
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first" if model_family != "tree" else None,
                    sparse_output=False,
                ),
            ),
        ]
    )
    columns = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numerical),
            (
                "binary",
                SimpleImputer(
                    strategy="most_frequent",
                    missing_values=pd.NA,
                    keep_empty_features=True,
                ),
                binary,
            ),
            ("categorical", category_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )
    steps = []
    if feature_set in {"engineered", "enhanced"}:
        engineer = (
            EnhancedTelcoFeatureEngineer
            if feature_set == "enhanced"
            else TelcoFeatureEngineer
        )
        steps.append(("engineering", engineer(config_path=config_path)))
    steps.append(("columns", columns))
    return Pipeline(steps)
