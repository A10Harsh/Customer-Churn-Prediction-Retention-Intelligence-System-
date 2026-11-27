"""Focused synthetic ANN architecture, stopping, and leakage tests."""

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.model_selection import train_test_split

from retention.models import ann_training
from retention.models.ann import (
    RetentionANN,
    positive_weight,
    predict_probabilities,
    train_network,
)
from retention.models.ann_training import cross_validate_ann
from retention.models.training import make_folds
from retention.utils.config import load_config


def synthetic_telco() -> tuple[pd.DataFrame, pd.Series]:
    base = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 6,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 40.0,
        "TotalCharges": 240.0,
    }
    rows = []
    for index in range(20):
        rows.append(
            {
                **base,
                "gender": "Male" if index % 2 else "Female",
                "tenure": index + 1,
                "TotalCharges": 40.0 * (index + 1),
            }
        )
    return pd.DataFrame(rows), pd.Series([0, 1] * 10)


def test_network_shape_loss_and_eval_probabilities() -> None:
    network = RetentionANN(5, (64, 32, 16), 0.2)
    X = torch.randn(4, 5)
    logits = network(X)
    assert logits.shape == (4,)
    assert network(torch.randn(1, 5)).shape == (1,)
    loss = torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.isfinite(loss)
    probabilities = predict_probabilities(network, X.numpy(), torch.device("cpu"))
    assert not network.training
    assert probabilities.shape == (4,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_weight_and_early_stopping() -> None:
    assert positive_weight(np.array([0, 0, 0, 1])) == 3
    with pytest.raises(ValueError, match="both classes"):
        positive_weight(np.array([0, 0]))
    settings = load_config("params.yaml")["ann_training"]
    settings.update(learning_rate=0.0, early_stopping_patience=2)
    X = np.random.default_rng(42).normal(size=(12, 5)).astype("float32")
    y = np.array([0, 1] * 6)
    model, history, best_epoch, weight = train_network(
        X, y, settings, torch.device("cpu"), 42, True, 10, X[:4], y[:4]
    )
    assert best_epoch == 1
    assert len(history) == 3
    assert weight == 1
    assert not model.training
    assert all(np.isfinite(row["train_loss"]) for row in history)
    assert all(np.isfinite(row["valid_loss"]) for row in history)


def test_inner_outer_separation_and_oof_coverage() -> None:
    X, y = synthetic_telco()
    X = pd.concat([X, X], ignore_index=True)
    y = pd.Series([0] * 30 + [1] * 10)
    config = load_config("params.yaml")
    config["ann_training"].update(max_epochs=2, early_stopping_patience=1)
    folds = make_folds(y, 2, 42)
    result = cross_validate_ann(
        X, y, "engineered", "weighted", config, folds, torch.device("cpu")
    )
    assert set(result["fold_ids"]) == {1, 2}
    assert len(result["oof_probabilities"]) == len(y)
    assert np.isfinite(result["oof_probabilities"]).all()
    for row, (outer_train, outer_valid) in zip(
        result["fold_metrics"].itertuples(), folds, strict=True
    ):
        inner_train, _ = train_test_split(
            outer_train,
            test_size=config["ann_training"]["inner_validation_size"],
            stratify=y.iloc[outer_train],
            random_state=42 + row.fold,
        )
        assert row.inner_train_rows + row.inner_validation_rows == len(outer_train)
        assert row.outer_validation_rows == len(outer_valid)
        assert row.inner_pos_weight == positive_weight(y.iloc[inner_train].to_numpy())
        assert row.refit_pos_weight == positive_weight(y.iloc[outer_train].to_numpy())
        assert row.best_epoch <= 2
    assert set(result["history"].phase) == {"selection", "refit"}


def test_synthetic_ann_cli_logging(tmp_path, monkeypatch) -> None:
    X, y = synthetic_telco()
    data = X.assign(customerID=[f"S{i}" for i in range(len(X))])
    data["Churn"] = y.map({0: "No", 1: "Yes"})
    development = tmp_path / "development.parquet"
    data.to_parquet(development)
    classical = tmp_path / "classical"
    classical.mkdir()
    pd.DataFrame(
        [{"name": "dummy__baseline__unweighted", "model": "dummy", "ap_mean": 0.5}]
    ).to_csv(classical / "leaderboard.csv", index=False)
    config = load_config("params.yaml")
    config["data"]["development_path"] = str(development)
    config["data"]["test_path"] = str(tmp_path / "reserved.parquet")
    config["validation"]["cv_folds"] = 2
    config["model_training"]["report_directory"] = str(classical)
    config["model_training"]["mlflow_tracking_uri"] = (
        f"sqlite:///{tmp_path / 'tracking.db'}"
    )
    config["ann_training"].update(
        max_epochs=2,
        early_stopping_patience=1,
        feature_modes=["baseline"],
        loss_strategies=["unweighted"],
        report_directory=str(tmp_path / "ann_reports"),
        mlflow_experiment="synthetic-ann-test",
    )
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ann_training, "_git_state", lambda: ("synthetic", True))
    plot = tmp_path / "plot.txt"
    plot.write_text("synthetic plot")
    monkeypatch.setattr(ann_training, "_plot_history", lambda *_: plot)
    monkeypatch.setattr(ann_training, "_plot_comparison", lambda *_: plot)

    board = ann_training.run_ann_experiments(str(config_path))
    assert len(board) == 1
    experiment = mlflow.get_experiment_by_name("synthetic-ann-test")
    runs = mlflow.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["params.fold_checksum"]
    assert "metrics.cv_mean_average_precision" in runs
    assert (tmp_path / "ann_reports" / "leaderboard.csv").is_file()
