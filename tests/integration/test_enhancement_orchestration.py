"""Exercise enhancement publication with synthetic OOF data and no estimator fits."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from retention.evaluation.metrics import binary_metrics
from retention.models import xgboost_enhancement as enhancement
from retention.utils.config import load_config


def test_enhancement_publishes_and_resumes_synthetic_study(tmp_path, monkeypatch):
    source_root = Path.cwd()
    config = copy.deepcopy(load_config("params.yaml"))
    y = np.array([0, 1] * 10)
    development = tmp_path / "data" / "processed" / "development.parquet"
    development.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "customerID": [f"S{i}" for i in range(20)],
            "tenure": np.arange(20),
            "Churn": np.where(y, "Yes", "No"),
        }
    ).to_parquet(development)
    config["data"]["development_path"] = str(development)
    config["data"]["test_path"] = str(tmp_path / "reserved.parquet")
    config["xgboost_enhancement"]["storage_path"] = str(tmp_path / "study.db")
    config["xgboost_enhancement"]["report_directory"] = str(tmp_path / "output")
    config["xgboost_enhancement"]["trials"] = 1
    config["model_training"]["mlflow_tracking_uri"] = (
        f"sqlite:///{tmp_path / 'tracking.db'}"
    )
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config))
    prior = tmp_path / "reports" / "generated" / "model_optimization"
    prior.mkdir(parents=True)
    labels = pd.Series(y)
    folds = enhancement.make_folds(labels, 5, 42)
    (prior / "selected_configuration.json").write_text(
        json.dumps(
            {
                "dataset_sha256": enhancement._checksum(development),
                "fold_sha256": enhancement._fold_hash(folds),
            }
        )
    )
    parameters = {
        "n_estimators": 200,
        "max_depth": 1,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": 1,
        "tree_method": "hist",
    }
    (prior / "best_hyperparameters.json").write_text(
        json.dumps(
            {
                model: {"parameters": parameters}
                for model in ["xgboost", "logistic_regression", "random_forest", "ann"]
            }
        )
    )
    model_names = ["xgboost", "logistic_regression", "random_forest", "ann"]
    pd.DataFrame(
        [
            {
                "model": model,
                "mean_average_precision": 0.5,
                "std_average_precision": 0.01,
                "feature_mode": "baseline",
                "strategy": "unweighted",
                "total_training_seconds": 1.0,
                "total_inference_seconds": 1.0,
            }
            for model in model_names
        ]
    ).to_csv(prior / "leaderboard.csv", index=False)
    old_metrics = pd.DataFrame({"fold": range(1, 6), "average_precision": [0.5] * 5})
    for model in model_names:
        old_metrics.to_csv(prior / f"{model}_fold_metrics.csv", index=False)
    initial_dir = (
        tmp_path
        / "reports"
        / "generated"
        / "model_training"
        / "xgboost__baseline__weighted"
    )
    initial_dir.mkdir(parents=True)
    (initial_dir / "configuration.json").write_text(
        json.dumps(
            {
                "mean": {
                    "average_precision": 0.5,
                    "training_seconds": 0.2,
                    "inference_seconds": 0.2,
                },
                "std": {"average_precision": 0.01},
                "hyperparameters": parameters,
            }
        )
    )
    old_metrics.to_csv(initial_dir / "fold_metrics.csv", index=False)
    for relative in [
        "src/retention/features/engineering.py",
        "src/retention/features/preprocessing.py",
        "src/retention/evaluation/threshold.py",
        "src/retention/models/optimization.py",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(enhancement, "validate_config", lambda _config: None)
    monkeypatch.setattr(enhancement, "_git_state", lambda: ("synthetic", True))
    scores = np.where(y, 0.8, 0.2)

    def result(ap):
        frame = pd.DataFrame(
            {
                "fold": range(1, 6),
                "average_precision": [ap] * 5,
                "training_seconds": [0.0] * 5,
                "inference_seconds": [0.0] * 5,
            }
        )
        return {
            "mean": {"average_precision": ap, "recall": 0.8, "precision": 0.6},
            "std": {"average_precision": 0.01},
            "pooled": binary_metrics(y, scores),
            "fold_metrics": frame,
            "fold_ids": np.repeat(np.arange(1, 6), 4),
            "oof_probabilities": scores,
        }

    monkeypatch.setattr(enhancement, "_prior_result", lambda *_: result(0.5))
    monkeypatch.setattr(
        enhancement,
        "_screen",
        lambda _out, name, *_args, **_kw: result(
            0.65 if name.startswith("extended_trial") else 0.4
        ),
    )
    monkeypatch.setattr(enhancement, "evaluate_xgboost", lambda *_a, **_k: result(0.65))
    monkeypatch.setattr(
        enhancement, "calibrated_oof", lambda *_a, **_k: (scores, scores)
    )
    monkeypatch.setattr(enhancement, "_old_scores", lambda _name: (y, scores))
    selected = enhancement.run_enhancement(str(config_path))
    assert selected["model"] == "xgboost_enhanced"
    assert selected["development_sha256"] == enhancement._checksum(development)
    output = Path(config["xgboost_enhancement"]["report_directory"])
    assert (output / "revised_selection.json").is_file()
    assert (output / "consolidated_leaderboard.csv").is_file()
    assert (output / "trial_history.csv").is_file()
    # A second pass uses the completed, compatible Optuna study.
    again = enhancement.run_enhancement(str(config_path))
    assert again["threshold"] == selected["threshold"]
    assert source_root != Path.cwd()  # temporary test CWD is active until teardown
