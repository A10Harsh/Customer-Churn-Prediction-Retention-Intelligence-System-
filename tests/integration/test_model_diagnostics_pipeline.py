"""Synthetic end-to-end diagnostics without development-model refits."""

import json

import numpy as np
import pandas as pd
import yaml

from retention.evaluation import diagnostic_extras, diagnostics
from retention.models.training import make_folds
from retention.utils.config import load_config
from tests.unit.test_model_training import synthetic_telco


def test_diagnostics_pipeline_publishes_synthetic_artifacts(tmp_path, monkeypatch):
    X, y = synthetic_telco()
    data = X.assign(customerID=[f"S{i}" for i in range(len(X))])
    data["Churn"] = y.map({0: "No", 1: "Yes"})
    development = tmp_path / "development.parquet"
    data.to_parquet(development)
    config = load_config("params.yaml")
    config["data"]["development_path"] = str(development)
    config["data"]["test_path"] = str(tmp_path / "reserved.parquet")
    config["model_training"]["mlflow_tracking_uri"] = (
        f"sqlite:///{tmp_path / 'diagnostics.db'}"
    )
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config))
    source = tmp_path / "frozen"
    source.mkdir()
    output = tmp_path / "reports"
    monkeypatch.setattr(diagnostics, "SOURCE", source)
    monkeypatch.setattr(diagnostics, "OUTPUT", output)
    monkeypatch.setattr(diagnostic_extras, "SOURCE", source)
    monkeypatch.setattr(diagnostic_extras, "OUTPUT", output)
    folds = make_folds(y, 5, 42)
    fold_ids = np.full(len(y), -1)
    for number, (_, valid) in enumerate(folds, 1):
        fold_ids[valid] = number
    scores = np.where(y.to_numpy(), 0.8, 0.2)
    selection = {
        "development_sha256": diagnostics._checksum(development),
        "fold_sha256": diagnostics._fold_hash(folds),
        "threshold": 0.5,
        "selected_metrics": {"tn": 10, "fp": 0, "fn": 0, "tp": 10},
        "parameters": {
            "n_estimators": 3,
            "max_depth": 1,
            "learning_rate": 0.1,
            "n_jobs": 1,
            "tree_method": "hist",
        },
        "scale_pos_weight": 1.0,
    }
    (source / "revised_selection.json").write_text(json.dumps(selection))
    (source / "calibration_oof.csv").write_text(
        pd.DataFrame(
            {
                "observed_churn": y,
                "raw_probability": scores,
                "sigmoid_probability": scores,
            }
        ).to_csv(index=False)
    )
    pd.DataFrame(
        {"fold": fold_ids, "observed_churn": y, "oof_probability": scores}
    ).to_csv(source / "extended_trial_35_oof.csv", index=False)
    prior = tmp_path / "reports" / "generated" / "model_optimization"
    prior.mkdir(parents=True)
    for model in ["logistic_regression", "random_forest", "xgboost", "ann"]:
        pd.DataFrame(
            {"fold": fold_ids, "observed_churn": y, "oof_probability": scores}
        ).to_csv(prior / f"{model}_oof.csv", index=False)
    monkeypatch.chdir(tmp_path)

    def fake_fit_fold(X, y, train, valid, *args, **kwargs):
        assert not set(train) & set(valid)
        return (
            None,
            np.where(y.iloc[valid], 0.8, 0.2),
            np.where(y.iloc[train], 0.8, 0.2),
        )

    monkeypatch.setattr(diagnostics, "fit_fold", fake_fit_fold)
    result = diagnostics.run(str(config_path))
    assert (result["baseline"]["tn"], result["baseline"]["fp"]) == (10, 0)
    final = diagnostic_extras.finish(str(config_path))
    assert final["overlap_and_explanation"]["shap_additivity"] is True
    assert (output / "report.html").is_file()
    assert (output / "segments.csv").is_file()
    assert (output / "ranking_capacity.png").is_file()
