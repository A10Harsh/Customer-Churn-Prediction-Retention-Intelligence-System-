"""Fast orchestration and resume checks without fitting or searching models."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from retention.evaluation.metrics import binary_metrics
from retention.models import optimization
from retention.utils.config import load_config


def test_optimization_publishes_and_resumes_frozen_synthetic_studies(
    tmp_path, monkeypatch
):
    config = copy.deepcopy(load_config("params.yaml"))
    development = tmp_path / "data" / "processed" / "development.parquet"
    development.parent.mkdir(parents=True)
    labels = np.array([0, 1] * 10)
    pd.DataFrame(
        {
            "customerID": [f"S{i}" for i in range(20)],
            "tenure": np.arange(20),
            "Churn": np.where(labels, "Yes", "No"),
        }
    ).to_parquet(development)
    config["data"]["development_path"] = str(development)
    config["data"]["test_path"] = str(tmp_path / "reserved.parquet")
    config["model_optimization"]["storage_path"] = str(tmp_path / "study.db")
    config["model_optimization"]["report_directory"] = str(tmp_path / "output")
    config["model_optimization"]["trial_budgets"] = dict.fromkeys(
        optimization.MODELS, 1
    )
    config["model_training"]["mlflow_tracking_uri"] = (
        f"sqlite:///{tmp_path / 'tracking.db'}"
    )
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config))
    for family, models in [
        ("model_training", optimization.MODELS[:-1]),
        ("ann_training", ("ann",)),
    ]:
        directory = tmp_path / "reports" / "generated" / family
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"model": model, "ap_mean": 0.4, "strategy": "unweighted"}
                for model in models
            ]
        ).to_csv(directory / "leaderboard.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(optimization, "_git_state", lambda: ("synthetic", True))
    calls = []

    def fake_evaluate(X, y, model, parameters, strategy, *_):
        calls.append(model)
        scores = np.where(y.to_numpy(), 0.8, 0.2)
        folds = optimization.make_folds(y, 5, 42)
        fold_ids = np.full(len(y), -1)
        for number, (_, valid) in enumerate(folds, 1):
            fold_ids[valid] = number
        fold_metrics = pd.DataFrame(
            {
                "fold": range(1, 6),
                "average_precision": [0.6] * 5,
                "training_seconds": [0.0] * 5,
                "inference_seconds": [0.0] * 5,
            }
        )
        return {
            "fold_metrics": fold_metrics,
            "fold_ids": fold_ids,
            "oof_probabilities": scores,
            "mean": {"average_precision": 0.6},
            "std": {"average_precision": 0.0},
            "pooled": binary_metrics(y, scores),
        }

    monkeypatch.setattr(optimization, "evaluate_candidate", fake_evaluate)
    monkeypatch.setattr(
        optimization,
        "calibrated_oof",
        lambda X, y, *_: (
            np.where(y.to_numpy(), 0.8, 0.2),
            np.where(y.to_numpy(), 0.8, 0.2),
        ),
    )
    selected = optimization.run_optimization(str(config_path))
    assert selected["model"] == "logistic_regression"
    assert selected["dataset_sha256"] == optimization._checksum(development)
    output = Path(config["model_optimization"]["report_directory"])
    assert (output / "selected_configuration.json").is_file()
    assert (output / "trial_history.csv").is_file()
    assert len(calls) == 8  # one trial and one best-candidate evaluation per model
    optimization.run_optimization(str(config_path))
    assert len(calls) == 12  # resumed studies make no new trial calls

    def fail_artifact_publication(_path):
        raise RuntimeError("artifact store unavailable")

    monkeypatch.setattr(optimization.mlflow, "log_artifacts", fail_artifact_publication)
    with pytest.raises(RuntimeError, match="artifact store unavailable"):
        optimization.run_optimization(str(config_path))
    assert optimization.mlflow.active_run() is None
