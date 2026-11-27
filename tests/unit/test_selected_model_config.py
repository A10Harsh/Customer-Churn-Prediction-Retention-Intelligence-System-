"""Guard the exact selected-model contract against accidental drift."""

import json
from pathlib import Path

import pytest
import yaml
from sklearn.base import clone

from retention.evaluation.calibration import apply_sigmoid, fit_sigmoid
from retention.features.preprocessing import make_preprocessor
from retention.models.factory import make_model
from retention.models.training import _checksum


def test_selected_configuration_matches_frozen_selection():
    frozen = yaml.safe_load(Path("configs/selected_model.yaml").read_text())
    assert frozen["schema_version"] == 1
    assert frozen["hyperparameters"]["n_estimators"] == 535
    assert frozen["hyperparameters"]["max_depth"] == 1
    assert frozen["classification"]["threshold"] == 0.3304499866713542
    assert frozen["development_integrity"]["sha256"] == (
        "28d12b92122ab94a8172be83e2bcf7e93b128a1bf3633c2b48075c426864da13"
    )
    source_path = Path("reports/generated/xgboost_enhancement/revised_selection.json")
    if source_path.is_file():
        source = json.loads(source_path.read_text())
        assert frozen["hyperparameters"] == source["parameters"]
        assert (
            frozen["class_weighting"]["scale_pos_weight"] == source["scale_pos_weight"]
        )
        assert frozen["classification"]["threshold"] == source["threshold"]
        assert frozen["development_integrity"]["fold_sha256"] == source["fold_sha256"]
    development = Path("data/processed/development.parquet")
    if development.is_file():
        assert frozen["development_integrity"]["sha256"] == _checksum(development)
    assert frozen["calibration"]["method"] == "sigmoid"
    assert frozen["target"]["positive_class"] == 1
    preprocessor = make_preprocessor("baseline", "tree")
    assert clone(preprocessor) is not preprocessor
    estimator = make_model(
        "xgboost",
        "weighted",
        42,
        frozen["hyperparameters"],
        frozen["class_weighting"]["scale_pos_weight"],
    )
    assert estimator.get_params()["n_estimators"] == 535
    calibrator = fit_sigmoid([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert apply_sigmoid(calibrator, [0.3])[0] < apply_sigmoid(calibrator, [0.7])[0]


def test_model_factory_rejects_invalid_weights_and_models():
    with pytest.raises(ValueError, match="positive fold weight"):
        make_model("xgboost", "weighted", 42, {}, 0)
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("unsupported", "unweighted", 42, {})
