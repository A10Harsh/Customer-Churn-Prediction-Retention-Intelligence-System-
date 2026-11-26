"""Leakage and compatibility checks for unfitted preprocessing factories."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from retention.features.preprocessing import make_preprocessor, split_features_target


def rows() -> pd.DataFrame:
    base: dict[str, object] = {
        "customerID": "A",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 6,
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
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 50.0,
        "TotalCharges": 120.0,
        "Churn": "Yes",
    }
    first = base.copy()
    second = {**base, "customerID": "B", "gender": "Male", "Churn": "No"}
    third = {
        **base,
        "customerID": "C",
        "tenure": 0,
        "TotalCharges": 0.0,
        "MonthlyCharges": np.nan,
        "gender": None,
        "Churn": "No",
    }
    return pd.DataFrame([first, second, third])


@pytest.mark.parametrize("feature_set", ["baseline", "engineered"])
@pytest.mark.parametrize("model_family", ["linear", "neural", "tree"])
def test_configurations_are_finite_cloneable_and_immutable(
    feature_set: str, model_family: str
) -> None:
    data = rows()
    original = data.copy(deep=True)
    X, y = split_features_target(data)
    X["gender"] = X["gender"].astype("string")
    pipeline = make_preprocessor(feature_set, model_family)
    clone_pipeline = clone(pipeline)
    transformed = clone_pipeline.fit_transform(X, y)
    transformed_with_metadata = clone(pipeline).fit_transform(data, y)
    names = clone_pipeline.get_feature_names_out()

    assert transformed.shape == (3, len(names))
    assert len(names) == len(set(names))
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(transformed, transformed_with_metadata)
    assert "customerID" not in " ".join(names)
    assert "Churn" not in " ".join(names)
    assert y.tolist() == [1, 0, 0]
    assert "customerID" not in X and "Churn" not in X
    assert "columns" in clone_pipeline.named_steps
    assert ("engineering" in clone_pipeline.named_steps) == (
        feature_set == "engineered"
    )
    pd.testing.assert_frame_equal(data, original)


def test_zero_tenure_is_imputed_and_binary_indicators_are_unscaled() -> None:
    X, y = split_features_target(rows())
    pipeline = make_preprocessor("engineered", "linear")
    transformed = pipeline.fit_transform(X, y)
    names = list(pipeline.get_feature_names_out())
    avg_index = names.index("numeric__avg_monthly_spend")
    binary_index = names.index("binary__is_month_to_month")
    assert np.isfinite(transformed[2, avg_index])
    assert set(transformed[:, binary_index]) == {1.0}
    assert "numeric__avg_monthly_spend" in names


def test_scaling_is_only_in_linear_or_neural_numeric_path() -> None:
    for family in ("linear", "neural", "tree"):
        pipeline = make_preprocessor(model_family=family)
        numeric = pipeline.named_steps["columns"].transformers[0][1]
        assert ("scaler" in numeric.named_steps) == (family != "tree")


def test_unknown_categorical_and_missing_values() -> None:
    train, _ = split_features_target(rows().iloc[:2])
    holdout = train.iloc[[0]].copy()
    holdout.loc[:, "gender"] = "Unseen"
    holdout.loc[:, "MonthlyCharges"] = np.nan
    for family in ("linear", "tree"):
        for feature_set in ("baseline", "engineered"):
            pipeline = make_preprocessor(feature_set, family).fit(train)
            result = pipeline.transform(holdout)
            assert np.isfinite(result).all()
            assert result.shape[1] == len(pipeline.get_feature_names_out())


def test_explicit_subset_and_invalid_selection() -> None:
    X, _ = split_features_target(rows())
    chosen = ["tenure", "SeniorCitizen", "Contract"]
    pipeline = make_preprocessor(selected_features=chosen).fit(X)
    names = list(pipeline.get_feature_names_out())
    assert "numeric__tenure" in names
    assert "binary__SeniorCitizen" in names
    assert all("MonthlyCharges" not in name for name in names)
    with pytest.raises(ValueError, match="Unknown selected features"):
        make_preprocessor(selected_features=["Churn"])
    with pytest.raises(ValueError, match="unique"):
        make_preprocessor(selected_features=["tenure", "tenure"])


def test_target_validation_and_training_fold_only_statistics() -> None:
    data = rows()
    bad = data.copy()
    bad.loc[0, "Churn"] = "Unknown"
    with pytest.raises(ValueError, match="No or Yes"):
        split_features_target(bad)

    X, y = split_features_target(data)
    train = X.iloc[:2].copy()
    validation = X.iloc[[2]].copy()
    validation.loc[:, "tenure"] = 999
    pipeline = make_preprocessor(selected_features=["tenure"])
    assert isinstance(pipeline, Pipeline)
    pipeline.fit(train, y.iloc[:2])
    median = (
        pipeline.named_steps["columns"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_[0]
    )
    assert median == 6
    assert pipeline.transform(validation).shape == (1, 1)
    assert median != validation["tenure"].iloc[0]


def test_single_customer_batch_and_independent_fitted_state() -> None:
    X, y = split_features_target(rows())
    first = make_preprocessor("engineered", "linear")
    second = make_preprocessor("engineered", "linear")
    first.fit(X.iloc[:2], y.iloc[:2])
    shifted = X.iloc[:2].copy()
    shifted["MonthlyCharges"] += 100
    second.fit(shifted, y.iloc[:2])

    first_numeric = first.named_steps["columns"].named_transformers_["numeric"]
    second_numeric = second.named_steps["columns"].named_transformers_["numeric"]
    monthly_index = list(first_numeric.feature_names_in_).index("MonthlyCharges")
    assert first_numeric.named_steps["imputer"].statistics_[monthly_index] == 50
    assert second_numeric.named_steps["imputer"].statistics_[monthly_index] == 150

    one = first.transform(X.iloc[[2]])
    batch = first.transform(X.iloc[[0, 2]])
    assert one.shape == (1, len(first.get_feature_names_out()))
    np.testing.assert_array_equal(one[0], batch[1])
    assert np.isfinite(one).all()


def test_stratified_fold_preprocessing_and_estimator_pipeline_cloning() -> None:
    X, y = split_features_target(pd.concat([rows()] * 4, ignore_index=True))
    estimator_pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor("engineered", "linear")),
            ("estimator", LogisticRegression()),
        ]
    )
    for train_index, valid_index in StratifiedKFold(
        n_splits=2, shuffle=True, random_state=42
    ).split(X, y):
        fold = clone(estimator_pipeline).named_steps["preprocess"]
        assert not hasattr(fold.named_steps["columns"], "transformers_")
        train_matrix = fold.fit_transform(X.iloc[train_index], y.iloc[train_index])
        valid_matrix = fold.transform(X.iloc[valid_index])
        assert train_matrix.shape[1] == valid_matrix.shape[1]
        assert valid_matrix.shape[1] == len(fold.get_feature_names_out())
        assert np.isfinite(train_matrix).all() and np.isfinite(valid_matrix).all()
