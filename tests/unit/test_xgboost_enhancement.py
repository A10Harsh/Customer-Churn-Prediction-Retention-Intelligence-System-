"""Focused synthetic checks for the Phase 6A+ additions."""

import numpy as np
import pandas as pd
import pytest

from retention.evaluation.threshold import (
    operating_points,
    select_constrained_recall,
    select_threshold,
)
from retention.features.engineering import EnhancedTelcoFeatureEngineer
from retention.features.preprocessing import make_preprocessor
from retention.models.training import make_folds
from retention.models.xgboost_enhancement import (
    _synthetic_valid,
    evaluate_xgboost,
    resample_training,
)
from retention.utils.config import load_config
from tests.unit.test_model_training import synthetic_telco


def test_enhanced_features_fit_charge_edges_only_on_training():
    X, _ = synthetic_telco()
    engineer = EnhancedTelcoFeatureEngineer().fit(X.iloc[:16])
    edges = engineer.charge_edges_.copy()
    valid = X.iloc[16:].copy()
    valid["MonthlyCharges"] = 1000.0
    result = engineer.transform(valid)
    np.testing.assert_array_equal(engineer.charge_edges_, edges)
    assert "contract_tenure" in result
    assert "payment_tenure" in result
    assert "internet_charge_band" in result
    assert "support_security_status" in result
    assert "monthly_charge_per_service" in result
    assert "new_customer" in result
    assert result.internet_charge_band.str.endswith("|3").all()
    pipeline = make_preprocessor("enhanced", "tree")
    assert pipeline.fit_transform(X.iloc[:16]).shape[1] > 20
    assert pipeline.transform(valid).shape[0] == 4


def test_synthetic_domain_validation_and_train_only_smote():
    X, _ = synthetic_telco()
    X = pd.concat([X, X.iloc[:10]], ignore_index=True)
    y = pd.Series([0] * 20 + [1] * 10)
    altered = X.iloc[:1].copy()
    altered["MultipleLines"] = "No phone service"
    assert not _synthetic_valid(altered).iloc[0]
    sampled_X, sampled_y, detail = resample_training(X, y, 1.0, 42)
    assert len(sampled_X) == len(sampled_y)
    assert (
        detail["synthetic_accepted"] + detail["synthetic_rejected"]
        == detail["synthetic_generated"]
    )
    assert _synthetic_valid(sampled_X).all()
    assert detail["effective_ratio"] <= 1.0
    with pytest.raises(ValueError, match="Identifiers"):
        resample_training(X.assign(customerID="x"), y, 1.0, 42)


def test_reduced_feature_pipeline_is_fold_local():
    X, y = synthetic_telco()
    config = load_config("params.yaml")
    folds = make_folds(y, 2, 42)
    parameters = {
        "n_estimators": 3,
        "max_depth": 2,
        "learning_rate": 0.05,
        "n_jobs": 1,
        "tree_method": "hist",
    }
    result = evaluate_xgboost(X, y, folds, config, "baseline", parameters, reduced=True)
    assert np.isfinite(result["oof_probabilities"]).all()
    assert set(result["fold_ids"]) == {1, 2}
    with pytest.raises(ValueError, match="cannot be combined"):
        evaluate_xgboost(
            X,
            y,
            folds,
            config,
            "baseline",
            parameters,
            weight="weighted",
            smote_ratio=0.5,
        )


def test_precision_constrained_policy_and_ties():
    table = operating_points([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    assert 0.5 in table.threshold.to_numpy()
    constrained = select_constrained_recall(table, 0.55)
    assert constrained["recall"] == 1
    assert constrained["precision"] >= 0.55
    assert select_constrained_recall(table, 1.0)["precision"] == 1
    ties = pd.DataFrame(
        {
            "recall": [0.8, 0.8, 0.8],
            "precision": [0.6, 0.7, 0.7],
            "f1": [0.7, 0.75, 0.75],
            "threshold": [0.2, 0.3, 0.4],
            "tp": [8, 8, 8],
            "fp": [4, 3, 3],
        }
    )
    assert select_constrained_recall(ties)["threshold"] == 0.4
    assert select_constrained_recall(ties, 0.9) is None
    assert select_threshold(table)["f1"] == 1
