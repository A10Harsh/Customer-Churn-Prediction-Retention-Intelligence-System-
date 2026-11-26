"""Focused synthetic checks for classical-model training primitives."""

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import clone

from retention.evaluation.metrics import binary_metrics, positive_probabilities
from retention.models import training
from retention.models.factory import make_estimator_pipeline, make_model
from retention.models.training import (
    cross_validate_experiment,
    experiment_specs,
    make_folds,
)
from retention.utils.config import load_config


def synthetic_telco() -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for index in range(20):
        rows.append(
            {
                "gender": "Female" if index % 2 else "Male",
                "SeniorCitizen": index % 2,
                "Partner": "Yes" if index % 3 else "No",
                "Dependents": "No",
                "tenure": index + 1,
                "PhoneService": "Yes",
                "MultipleLines": "Yes" if index % 2 else "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes" if index % 2 else "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes" if index % 3 else "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month" if index % 2 else "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 30.0 + index,
                "TotalCharges": 30.0 * (index + 1),
            }
        )
    return pd.DataFrame(rows), pd.Series([0, 1] * 10)


def test_metrics_and_positive_probability() -> None:
    scores = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert scores["average_precision"] == 1
    assert scores["roc_auc"] == 1
    assert (scores["tn"], scores["fp"], scores["fn"], scores["tp"]) == (2, 0, 0, 2)
    zeros = binary_metrics([0, 1], [0.1, 0.1])
    assert zeros["precision"] == zeros["recall"] == zeros["f1"] == 0
    with pytest.raises(ValueError, match="Both binary classes"):
        binary_metrics([0, 0], [0.1, 0.2])

    X, y = synthetic_telco()
    pipeline = make_estimator_pipeline("dummy", "baseline", "unweighted", 42, {})
    pipeline.fit(X, y)
    assert positive_probabilities(pipeline, X.iloc[:2]).shape == (2,)


def test_factory_configurations_and_fold_weight() -> None:
    config = load_config("params.yaml")
    assert len(experiment_specs(config)) == 13
    assert make_model("dummy", "unweighted", 42, {}).strategy == "prior"
    weighted = make_model(
        "xgboost", "weighted", 42, config["model_training"]["models"]["xgboost"], 3.0
    )
    assert weighted.scale_pos_weight == 3.0
    with pytest.raises(ValueError, match="fold weight"):
        make_model("xgboost", "weighted", 42, {}, None)
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("unknown", "unweighted", 42, {})


def test_repeated_folds_oof_coverage_and_clone() -> None:
    X, y = synthetic_telco()
    config = load_config("params.yaml")
    folds = make_folds(y, 5, 42)
    repeated = make_folds(y, 5, 42)
    for (_, validation), (_, again) in zip(folds, repeated, strict=True):
        np.testing.assert_array_equal(validation, again)
    pipeline = make_estimator_pipeline("dummy", "engineered", "unweighted", 42, {})
    assert clone(pipeline) is not pipeline
    result = cross_validate_experiment(
        X, y, ("dummy", "engineered", "unweighted"), config, folds
    )
    assert len(result["fold_metrics"]) == 5
    assert set(result["fold_ids"]) == {1, 2, 3, 4, 5}
    assert np.isfinite(result["oof_probabilities"]).all()
    assert result["fold_metrics"]["validation_rows"].sum() == len(y)


def test_weighted_logistic_and_forest_factory() -> None:
    config = load_config("params.yaml")["model_training"]["models"]
    for model in ("logistic_regression", "random_forest"):
        estimator = make_model(model, "weighted", 42, config[model])
        assert estimator.class_weight == "balanced"


def test_xgboost_weight_uses_each_training_fold() -> None:
    X, _ = synthetic_telco()
    y = pd.Series([0] * 15 + [1] * 5)
    config = load_config("params.yaml")
    config["model_training"]["models"]["xgboost"]["n_estimators"] = 2
    folds = make_folds(y, 2, 42)
    result = cross_validate_experiment(
        X, y, ("xgboost", "baseline", "weighted"), config, folds
    )
    expected = [
        float((y.iloc[train] == 0).sum() / (y.iloc[train] == 1).sum())
        for train, _ in folds
    ]
    assert result["fold_metrics"]["scale_pos_weight"].tolist() == expected


def test_local_mlflow_logging_with_synthetic_data(tmp_path, monkeypatch) -> None:
    X, y = synthetic_telco()
    data = X.assign(customerID=[f"S{i}" for i in range(len(X))])
    data["Churn"] = y.map({0: "No", 1: "Yes"})
    development = tmp_path / "development.parquet"
    data.to_parquet(development)
    config = load_config("params.yaml")
    config["data"]["development_path"] = str(development)
    config["data"]["test_path"] = str(tmp_path / "reserved.parquet")
    config["validation"]["cv_folds"] = 2
    config["model_training"]["report_directory"] = str(tmp_path / "reports")
    config["model_training"]["mlflow_tracking_uri"] = (
        f"sqlite:///{tmp_path / 'tracking.db'}"
    )
    config["model_training"]["mlflow_experiment"] = "synthetic-training-test"
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "_git_state", lambda: ("synthetic", True))
    monkeypatch.setattr(
        training,
        "experiment_specs",
        lambda _: [("dummy", "baseline", "unweighted")],
    )
    monkeypatch.setattr(training, "_plot_summary", lambda *_: [])

    board = training.run_training(str(config_path))
    assert len(board) == 1
    experiment = mlflow.get_experiment_by_name("synthetic-training-test")
    runs = mlflow.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["params.development_sha256"]
    assert "metrics.cv_mean_average_precision" in runs
    assert (tmp_path / "reports" / "leaderboard.csv").is_file()
