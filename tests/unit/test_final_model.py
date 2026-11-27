"""Synthetic contracts for the frozen final inference path."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from retention.evaluation.calibration import fit_sigmoid
from retention.evaluation.final_evaluation import bootstrap_intervals
from retention.models.final_training import validate_selection
from retention.models.predictor import ChurnPredictor, load_predictor


def _predictor():
    X = pd.DataFrame(
        {"tenure": [1, 2, 3, 4, 5, 6], "Contract": ["Monthly", "Annual"] * 3}
    )
    y = [0, 0, 0, 1, 1, 1]
    pipe = Pipeline([("scale", StandardScaler()), ("estimator", LogisticRegression())])
    pipe.fit(X[["tenure"]], y)
    calibrator = fit_sigmoid([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], y)
    return ChurnPredictor(
        NumericPipeline(pipe),
        calibrator,
        0.4,
        tuple(X.columns),
        ("tenure",),
        {"Contract": ("Monthly", "Annual")},
        {},
    )


class NumericPipeline:
    classes_ = np.array([0, 1])

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_proba(self, frame):
        return self.pipeline.predict_proba(frame[["tenure"]])


def test_single_batch_threshold_serialization_and_input_validation(tmp_path):
    model = _predictor()
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = load_predictor(path)
    one = {"tenure": 3, "Contract": "Monthly"}
    assert loaded.predict(one).equals(loaded.predict(pd.DataFrame([one])))
    frame = loaded.predict(one)
    assert frame.predicted_churn.iloc[0] == int(frame.churn_probability.iloc[0] >= 0.4)
    with pytest.raises(ValueError, match="Unsupported category"):
        loaded.predict({"tenure": 3, "Contract": "Unknown"})
    with pytest.raises(ValueError, match="Nonfinite"):
        loaded.predict({"tenure": np.inf, "Contract": "Monthly"})
    with pytest.raises(ValueError, match="schema mismatch"):
        loaded.predict({"tenure": 3})


def test_bootstrap_reproducible_and_counts():
    y = np.array([0, 1] * 10)
    p = np.linspace(0.05, 0.95, len(y))
    first = bootstrap_intervals(y, p, 0.4, seed=7, resamples=40)
    assert first == bootstrap_intervals(y, p, 0.4, seed=7, resamples=40)
    assert first["precision"]["valid_resamples"] == 40


def test_selection_rejects_changed_hyperparameters():
    import json
    from pathlib import Path

    import yaml

    from retention.models.training import make_folds

    frozen = yaml.safe_load(Path("configs/selected_model.yaml").read_text())
    source = json.loads(
        Path("reports/generated/xgboost_enhancement/revised_selection.json").read_text()
    )
    frozen["hyperparameters"]["n_estimators"] += 1
    labels = pd.read_parquet("data/processed/development.parquet").Churn.map(
        {"No": 0, "Yes": 1}
    )
    folds = make_folds(labels, 5, 42)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_selection(
            frozen, source, Path("data/processed/development.parquet"), folds
        )


def test_dvc_stage_boundaries():
    import yaml

    stages = yaml.safe_load(open("dvc.yaml"))["stages"]
    training = stages["final_train"]["deps"]
    evaluation = stages["final_evaluate"]["deps"]
    assert "${data.test_path}" not in training
    assert "${data.development_path}" in training
    assert "${data.test_path}" in evaluation
    assert "artifacts/final_model/readiness.json" in evaluation


def test_synthetic_evaluation_governance(tmp_path, monkeypatch):
    import json

    from retention.evaluation import final_evaluation as final

    model = tmp_path / "model.joblib"
    model.write_bytes(b"synthetic model")
    development = tmp_path / "development.parquet"
    development.write_bytes(b"synthetic development")
    test = tmp_path / "test.parquet"
    test.write_bytes(b"synthetic test")
    selection = tmp_path / "selected_model.yaml"
    selection.write_text("synthetic selection")
    manifest = tmp_path / "readiness.json"
    manifest.write_text(
        json.dumps(
            {
                "readiness": "passed",
                "model_sha256": "model",
                "configuration_sha256": "selection",
                "development_sha256": "development",
            }
        )
    )
    monkeypatch.setattr(final, "ARTIFACT", model)
    monkeypatch.setattr(final, "MANIFEST", manifest)
    monkeypatch.setattr(final, "GOVERNANCE", tmp_path / "guard.json")
    monkeypatch.setattr(
        final,
        "_checksum",
        lambda path: {
            model: "model",
            development: "development",
            test: "test",
            selection: "selection",
        }[path],
    )
    monkeypatch.setattr(
        final,
        "load_config",
        lambda _: {
            "data": {"development_path": str(development), "test_path": str(test)},
            "project": {"random_state": 42},
        },
    )
    real_path = final.Path
    monkeypatch.setattr(
        final,
        "Path",
        lambda value: (
            selection if value == "configs/selected_model.yaml" else real_path(value)
        ),
    )
    data = pd.DataFrame(
        {
            "customerID": [f"fake-{i}" for i in range(1409)],
            "Churn": ["Yes" if i % 3 == 0 else "No" for i in range(1409)],
            "tenure": np.arange(1409) % 73,
            "Contract": ["Month-to-month"] * 1409,
            "InternetService": ["DSL"] * 1409,
            "gender": ["Female"] * 1409,
            "SeniorCitizen": [0] * 1409,
            "Partner": ["No"] * 1409,
            "Dependents": ["No"] * 1409,
        }
    )
    monkeypatch.setattr(final.pd, "read_parquet", lambda _: data)

    class FakeModel:
        threshold = 0.33
        metadata = {
            "configuration_sha256": "selection",
            "development_sha256": "development",
        }

        def predict(self, frame):
            p = np.where(np.arange(len(frame)) % 3 == 0, 0.7, 0.2)
            return pd.DataFrame(
                {"churn_probability": p, "predicted_churn": (p >= 0.33).astype(int)}
            )

    monkeypatch.setattr(final, "load_predictor", lambda _: FakeModel())
    original_bootstrap = final.bootstrap_intervals
    monkeypatch.setattr(
        final,
        "bootstrap_intervals",
        lambda y, p, threshold, seed: original_bootstrap(y, p, threshold, seed, 20),
    )
    result = final.evaluate(output=tmp_path / "report")
    assert result["primary"]["precision"] == 1
    assert (tmp_path / "guard.json").exists()
    assert len(list((tmp_path / "report").glob("*.png"))) == 4
    assert final.evaluate(output=tmp_path / "report") == result
    manifest.write_text(
        json.dumps({**json.loads(manifest.read_text()), "model_sha256": "changed"})
    )
    with pytest.raises(ValueError, match="readiness"):
        final.evaluate(output=tmp_path / "report")


def test_registry_reuses_identical_candidate(tmp_path, monkeypatch):
    import json

    from retention.models import final_registry as registry

    artifact = tmp_path / "predictor.joblib"
    artifact.write_bytes(b"synthetic")
    manifest = tmp_path / "readiness.json"
    manifest.write_text(
        json.dumps(
            {
                "readiness": "passed",
                "model_sha256": "same-model",
                "configuration_sha256": "same-config",
            }
        )
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({"fingerprints": {"model_sha256": "same-model"}}))
    monkeypatch.setattr(registry, "ARTIFACT", artifact)
    monkeypatch.setattr(registry, "MANIFEST", manifest)
    monkeypatch.setattr(registry, "EVALUATION", evaluation)
    monkeypatch.setattr(registry, "_checksum", lambda _: "same-model")
    monkeypatch.setattr(
        registry,
        "load_config",
        lambda _: {"model_training": {"mlflow_tracking_uri": "sqlite:///synthetic.db"}},
    )

    class Version:
        tags = {"model_sha256": "same-model", "configuration_sha256": "same-config"}
        version = "7"
        run_id = "synthetic-run"

    class Client:
        alias = None

        def search_model_versions(self, _):
            return [Version()]

        def set_registered_model_alias(self, name, alias, version):
            self.alias = (name, alias, version)

    client = Client()
    monkeypatch.setattr(registry.mlflow, "set_tracking_uri", lambda _: None)
    monkeypatch.setattr(registry.mlflow, "MlflowClient", lambda: client)
    result = registry.register()
    assert result == {
        "run_id": "synthetic-run",
        "version": "7",
        "alias": "candidate",
        "reused": True,
    }
    assert client.alias == ("customer-churn-predictor", "candidate", "7")


def test_synthetic_final_training_is_development_only(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from sklearn.dummy import DummyClassifier

    from retention.models import final_training as final
    from retention.models.training import make_folds

    development = tmp_path / "development.parquet"
    development.write_bytes(b"synthetic development")
    test = tmp_path / "test.parquet"
    test.write_bytes(b"synthetic held-out, never read")
    X = pd.DataFrame(
        {"tenure": np.arange(5634) % 72, "SeniorCitizen": np.arange(5634) % 2}
    )
    y = pd.Series(np.arange(5634) % 2, dtype="int8")
    folds = make_folds(y, 5, 42)
    fold_ids = np.full(len(y), -1)
    for number, (_, valid) in enumerate(folds, 1):
        fold_ids[valid] = number
    raw_oof = pd.DataFrame(
        {
            "fold": fold_ids,
            "observed_churn": y,
            "oof_probability": np.where(y == 1, 0.7, 0.3),
        }
    )
    artifact = tmp_path / "model.joblib"
    manifest = tmp_path / "readiness.json"
    monkeypatch.setattr(final, "ARTIFACT", artifact)
    monkeypatch.setattr(final, "MANIFEST", manifest)
    monkeypatch.setattr(final, "_git_state", lambda: ("synthetic-commit", False))
    monkeypatch.setattr(final, "validate_selection", lambda *args: None)
    monkeypatch.setattr(
        final,
        "load_config",
        lambda _: {
            "data": {"development_path": str(development), "test_path": str(test)},
            "eda": {"numerical_features": ["tenure"]},
        },
    )
    monkeypatch.setattr(final, "split_features_target", lambda data, _: (X, y))
    monkeypatch.setattr(
        final,
        "make_estimator_pipeline",
        lambda *args: DummyClassifier(strategy="prior"),
    )
    monkeypatch.setattr(final.pd, "read_csv", lambda _: raw_oof)
    accessed = []

    def read_parquet(path):
        accessed.append(path)
        assert path == development
        return pd.DataFrame(index=range(5634))

    monkeypatch.setattr(final.pd, "read_parquet", read_parquet)
    result = final.train()
    assert accessed == [development]
    assert result["readiness"] == "passed"
    assert result["development_rows"] == 5634
    assert json.loads(manifest.read_text())["model_sha256"] == final._checksum(artifact)
    assert Path(result["artifact"]) == artifact
