"""Final development-only fit and readiness gate for the frozen XGBoost model."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml

from retention.evaluation.calibration import fit_sigmoid
from retention.features.preprocessing import split_features_target
from retention.models.factory import make_estimator_pipeline
from retention.models.optimization import _fold_hash
from retention.models.predictor import ChurnPredictor, load_predictor
from retention.models.training import _checksum, _git_state, make_folds
from retention.utils.config import load_config

ARTIFACT = Path("artifacts/final_model/predictor.joblib")
MANIFEST = Path("artifacts/final_model/readiness.json")
RAW_OOF = Path("reports/generated/xgboost_enhancement/extended_trial_35_oof.csv")
SELECTION = Path("reports/generated/xgboost_enhancement/revised_selection.json")


def validate_selection(frozen: dict, source: dict, development: Path, folds) -> None:
    if (
        frozen["schema_version"] != 1
        or frozen["model_family"] != "xgboost"
        or frozen["feature_mode"] != "baseline"
        or frozen["class_weighting"]["strategy"] != "custom_scale_pos_weight"
        or frozen["calibration"]["method"] != "sigmoid"
        or frozen["calibration"]["input_score"]
        != "XGBClassifier.predict_proba[:, positive_class]"
        or frozen["target"]["positive_class"] != 1
        or frozen["target"]["positive_label"] != "Yes"
    ):
        raise ValueError("Unsupported frozen model contract")
    pairs = [
        (frozen["hyperparameters"], source["parameters"]),
        (frozen["class_weighting"]["scale_pos_weight"], source["scale_pos_weight"]),
        (frozen["classification"]["threshold"], source["threshold"]),
        (frozen["development_integrity"]["fold_sha256"], source["fold_sha256"]),
        (frozen["development_integrity"]["sha256"], _checksum(development)),
        (frozen["development_integrity"]["fold_sha256"], _fold_hash(folds)),
    ]
    if any(left != right for left, right in pairs):
        raise ValueError("Frozen selection or development fingerprint mismatch")


def train(
    config_path: str = "params.yaml",
    selection_path: str = "configs/selected_model.yaml",
) -> dict:
    config = load_config(config_path)
    frozen_path = Path(selection_path)
    frozen = yaml.safe_load(frozen_path.read_text())
    development = Path(config["data"]["development_path"])
    if (
        development.name != "development.parquet"
        or development.resolve() == Path(config["data"]["test_path"]).resolve()
    ):
        raise ValueError("Training must use development.parquet only")
    data = pd.read_parquet(development)
    if len(data) != 5634:
        raise ValueError("Frozen development row count mismatch")
    X, y = split_features_target(data, config_path)
    seed = frozen["development_integrity"]["seed"]
    folds = make_folds(y, frozen["development_integrity"]["folds"], seed)
    source = json.loads(SELECTION.read_text())
    validate_selection(frozen, source, development, folds)
    oof = pd.read_csv(RAW_OOF)
    expected_fold = np.full(len(y), -1)
    for number, (_, valid) in enumerate(folds, 1):
        expected_fold[valid] = number
    if (
        len(oof) != len(y)
        or not np.array_equal(oof["fold"].to_numpy(), expected_fold)
        or not np.array_equal(oof["observed_churn"].to_numpy(), y.to_numpy())
        or not np.isfinite(oof["oof_probability"]).all()
        or not oof["oof_probability"].between(0, 1).all()
    ):
        raise ValueError("Saved raw OOF scores do not match frozen development folds")
    calibrator = fit_sigmoid(oof["oof_probability"].to_numpy(), y.to_numpy())
    pipeline = make_estimator_pipeline(
        "xgboost",
        "baseline",
        "weighted",
        seed,
        frozen["hyperparameters"],
        frozen["class_weighting"]["scale_pos_weight"],
        config_path,
    )
    pipeline.fit(X, y)
    numeric = tuple(config["eda"]["numerical_features"] + ["SeniorCitizen"])
    categories = {
        name: tuple(X[name].dropna().unique().tolist())
        for name in X
        if name not in numeric
    }
    commit, dirty = _git_state()
    metadata = {
        "configuration_sha256": _checksum(frozen_path),
        "development_sha256": _checksum(development),
        "fold_sha256": _fold_hash(folds),
        "raw_oof_sha256": _checksum(RAW_OOF),
        "calibration_source": str(RAW_OOF),
        "calibration_score": frozen["calibration"]["input_score"],
        "calibration_coef": float(calibrator.coef_[0, 0]),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "seed": seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "versions": {
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
        },
    }
    predictor = ChurnPredictor(
        pipeline,
        calibrator,
        frozen["classification"]["threshold"],
        tuple(X.columns),
        numeric,
        categories,
        metadata,
    )
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, ARTIFACT)
    reloaded = load_predictor(ARTIFACT)
    sample = X.iloc[:5]
    batch = reloaded.predict(sample)
    single = reloaded.predict(sample.iloc[0].to_dict())
    if not np.allclose(
        batch.iloc[0].to_numpy(dtype=float), single.iloc[0].to_numpy(dtype=float)
    ):
        raise ValueError("Single and batch inference disagree")
    if not np.allclose(
        batch.to_numpy(dtype=float), predictor.predict(sample).to_numpy(dtype=float)
    ):
        raise ValueError("Serialization changed predictions")
    if not np.array_equal(
        batch.predicted_churn.to_numpy(),
        (batch.churn_probability.to_numpy() >= predictor.threshold).astype(int),
    ):
        raise ValueError("Threshold application failed")
    manifest = {
        **metadata,
        "model_sha256": _checksum(ARTIFACT),
        "readiness": "passed",
        "development_rows": len(X),
        "artifact": str(ARTIFACT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--selection", default="configs/selected_model.yaml")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.selection), indent=2))


if __name__ == "__main__":
    main()
