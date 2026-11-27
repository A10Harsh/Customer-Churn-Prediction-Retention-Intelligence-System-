"""Focused checks for segment definitions and saved-OOF integrity."""

import numpy as np
import pandas as pd
import pytest

from retention.evaluation.diagnostics import fit_fold, safe_metrics, verify_alignment


def test_segment_metrics_and_undefined_rates():
    result = safe_metrics([1, 1, 0, 0], [0.8, 0.2, 0.7, 0.1], 0.5)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["fnr"] == result["fpr"] == 0.5
    assert result["fnr_low"] < 0.5 < result["fnr_high"]
    one_class = safe_metrics([0, 0], [0.1, 0.2], 0.5)
    assert np.isnan(one_class["ap"])
    assert np.isnan(one_class["fnr"])


def test_alignment_rejects_changed_labels(tmp_path, monkeypatch):
    data = pd.DataFrame({"Churn": ["No", "Yes"], "customerID": ["a", "b"]})
    path = tmp_path / "development.parquet"
    path.write_bytes(b"development")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    selection = {"development_sha256": digest, "fold_sha256": "fold"}
    monkeypatch.setattr("retention.evaluation.diagnostics._fold_hash", lambda _: "fold")
    calibration = pd.DataFrame(
        {"observed_churn": [0, 0], "raw_probability": [0.1, 0.2]}
    )
    raw = pd.DataFrame(
        {"observed_churn": [0, 1], "oof_probability": [0.1, 0.2], "fold": [1, 2]}
    )
    folds = [(np.array([1]), np.array([0])), (np.array([0]), np.array([1]))]
    with pytest.raises(ValueError, match="row aligned"):
        verify_alignment(data, selection, calibration, raw, folds, path)


def test_fit_fold_excludes_outer_validation_rows(monkeypatch):
    X = pd.DataFrame({"tenure": [1, 2, 3, 4], "Contract": ["a"] * 4})
    y = pd.Series([0, 1, 0, 1])
    seen = {}

    class Predictor:
        def predict_proba(self, rows):
            return np.tile([0.7, 0.3], (len(rows), 1))

    def fake_fit(train_X, train_y, *_args):
        seen["training_indices"] = train_X.index.tolist()
        seen["training_labels"] = train_y.index.tolist()
        return Predictor(), {}, None

    monkeypatch.setattr("retention.evaluation.diagnostics.fit_xgb_pipeline", fake_fit)
    fit_fold(X, y, np.array([0, 1, 2]), np.array([3]), {}, 2.0, 42)
    assert seen == {"training_indices": [0, 1, 2], "training_labels": [0, 1, 2]}
