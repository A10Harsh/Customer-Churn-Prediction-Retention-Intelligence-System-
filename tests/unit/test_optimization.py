"""Small Phase 6A checks without running the full search."""

import copy

import numpy as np
import optuna
import pandas as pd
import pytest

from retention.evaluation.calibration import (
    apply_sigmoid,
    fit_sigmoid,
    validate_probabilities,
)
from retention.evaluation.threshold import select_threshold, threshold_table
from retention.models.optimization import (
    MODELS,
    _fold_hash,
    _study,
    calibrated_oof,
    suggest_parameters,
    validate_config,
)
from retention.models.training import make_folds
from retention.utils.config import load_config


def test_config_and_reproducible_folds():
    config = load_config("params.yaml")
    validate_config(config)
    changed = copy.deepcopy(config)
    changed["validation"]["cv_folds"] = 4
    with pytest.raises(ValueError):
        validate_config(changed)
    y = pd.Series([0, 1] * 20)
    assert _fold_hash(make_folds(y, 5, 42)) == _fold_hash(make_folds(y, 5, 42))
    assert _fold_hash(make_folds(y, 5, 42)) != _fold_hash(make_folds(y, 5, 43))


@pytest.mark.parametrize("model", MODELS)
def test_search_spaces_are_usable(model):
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))

    def objective(trial):
        parameters, strategy = suggest_parameters(trial, model)
        assert strategy in {"weighted", "unweighted"}
        assert parameters
        return 0.5

    study.optimize(objective, n_trials=1)
    assert study.best_value == 0.5


def test_persistent_study_rejects_changed_data(tmp_path):
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    metadata = {"seed": 42, "dataset_sha256": "a"}
    study = _study("ann", {}, metadata, storage)
    study.optimize(lambda _: 0.5, n_trials=1)
    assert len(_study("ann", {}, metadata, storage).trials) == 1
    with pytest.raises(ValueError, match="Incompatible"):
        _study("ann", {}, {**metadata, "dataset_sha256": "b"}, storage)


def test_failed_trial_resumes_and_seed_change_is_rejected(tmp_path):
    storage = f"sqlite:///{tmp_path / 'interrupted.db'}"
    metadata = {"seed": 42, "dataset_sha256": "synthetic"}
    study = _study("xgboost", {}, metadata, storage)

    def interrupted(_trial):
        raise RuntimeError("interrupted")

    study.optimize(interrupted, n_trials=1, catch=(RuntimeError,))
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    resumed = _study("xgboost", {}, metadata, storage)
    resumed.optimize(lambda _trial: 0.6, n_trials=1)
    assert sum(t.state == optuna.trial.TrialState.COMPLETE for t in resumed.trials) == 1
    with pytest.raises(ValueError, match="Incompatible"):
        _study("xgboost", {}, {**metadata, "seed": 43}, storage)


def test_invalid_optimization_configurations():
    base = load_config("params.yaml")
    for change in [
        {"search_space_version": 99},
        {"trial_budgets": {"ann": 0}},
    ]:
        config = copy.deepcopy(base)
        config["model_optimization"].update(change)
        with pytest.raises(ValueError):
            validate_config(config)


def test_calibration_validation_and_application():
    for bad in ([float("nan")], [-0.1], [[0.1, 0.2]], []):
        with pytest.raises(ValueError):
            validate_probabilities(bad)
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    calibrated = apply_sigmoid(fit_sigmoid(probabilities, labels), probabilities)
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert calibrated[0] < calibrated[-1]


def test_calibration_outer_fold_isolation(monkeypatch):
    import retention.models.optimization as optimization

    X = pd.DataFrame({"row": range(40)})
    y = pd.Series([0, 1] * 20)
    folds = make_folds(y, 5, 42)
    seen = []

    def fake_fit(X, y, train_idx, valid_idx, *args):
        assert not set(train_idx) & set(valid_idx)
        seen.append((set(train_idx), set(valid_idx)))
        return np.asarray([0.2 if y.iloc[i] == 0 else 0.8 for i in valid_idx])

    monkeypatch.setattr(optimization, "_fit_predict", fake_fit)
    raw, calibrated = calibrated_oof(
        X,
        y,
        "logistic_regression",
        {},
        "unweighted",
        {"project": {"random_state": 42}},
        folds,
        "params.yaml",
    )
    assert np.isfinite(raw).all() and np.isfinite(calibrated).all()
    for outer_number, (train, valid) in enumerate(folds):
        for inner_call in seen[outer_number * 6 : outer_number * 6 + 5]:
            assert inner_call[0] <= set(train)
            assert inner_call[1] <= set(train)
            assert not inner_call[1] & set(valid)


def test_threshold_tie_breaking_and_reference():
    table = threshold_table([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    assert 0.5 in table.threshold.to_numpy()
    assert select_threshold(table)["f1"] == 1
    ties = pd.DataFrame(
        {"f1": [0.8, 0.8, 0.8], "recall": [0.7, 0.8, 0.8], "threshold": [0.9, 0.4, 0.6]}
    )
    assert select_threshold(ties)["threshold"] == 0.6
    with pytest.raises(ValueError):
        threshold_table([0, 1], [0.1, float("inf")])
