"""Development-only XGBoost enhancement after the Phase 6A baseline."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from retention.data.schemas import (
    internet_service_consistency,
    phone_service_consistency,
)
from retention.evaluation.calibration import calibration_metrics, reliability_points
from retention.evaluation.metrics import binary_metrics, positive_probabilities
from retention.evaluation.threshold import (
    operating_points,
    select_constrained_recall,
    select_threshold,
)
from retention.features.engineering import TelcoFeatureEngineer
from retention.features.preprocessing import make_preprocessor, split_features_target
from retention.models.factory import make_model
from retention.models.optimization import _fold_hash, _json, _study, calibrated_oof
from retention.models.training import (
    SCORES,
    _checksum,
    _git_state,
    cross_validate_experiment,
    make_folds,
)
from retention.utils.config import load_config

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
FAMILIES = (
    "xgboost_initial",
    "xgboost_optimized",
    "logistic_regression",
    "random_forest",
    "ann",
)


def validate_config(config: dict) -> None:
    settings = config["xgboost_enhancement"]
    if config["validation"]["cv_folds"] != 5:
        raise ValueError("Five established folds are required.")
    if settings["search_space_version"] != 2 or settings["trials"] != 50:
        raise ValueError("Expected Phase 6A+ search version 2 and 50 trials.")
    if not 0 < settings["minimum_precision"] <= 1:
        raise ValueError("Precision floor must be in (0, 1].")
    path = Path(config["data"]["development_path"])
    if (
        path.name != "development.parquet"
        or path.resolve() == Path(config["data"]["test_path"]).resolve()
    ):
        raise ValueError("Only development.parquet is allowed.")


def _synthetic_valid(frame: pd.DataFrame) -> pd.Series:
    """Check sampled raw records against the existing service constraints."""
    tenure = pd.to_numeric(frame["tenure"], errors="coerce")
    monthly = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
    total = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    valid = (
        phone_service_consistency(frame)
        & internet_service_consistency(frame)
        & tenure.between(0, 72)
        & tenure.mod(1).eq(0)
        & monthly.ge(0)
        & total.ge(0)
        & (~tenure.eq(0) | total.eq(0))
        & np.isfinite(tenure)
        & np.isfinite(monthly)
        & np.isfinite(total)
    )
    for column in ("Contract", "PaymentMethod", "InternetService"):
        valid &= frame[column].notna()
    return valid


def resample_training(X: pd.DataFrame, y: pd.Series, ratio: float, seed: int):
    """Apply SMOTE-NC to training rows, then reject invalid synthetic rows."""
    if ratio not in {0.5, 0.75, 1.0}:
        raise ValueError("Unsupported SMOTE-NC ratio.")
    if X.columns.isin(["customerID", "Churn"]).any():
        raise ValueError("Identifiers and labels cannot be resampling predictors.")
    clean = X.reset_index(drop=True).copy()
    labels = y.reset_index(drop=True).copy()
    if not _synthetic_valid(clean).all():
        raise ValueError("Original training records violate domain constraints.")
    scaler = StandardScaler()
    scaled = clean.copy()
    scaled[NUMERIC] = scaler.fit_transform(clean[NUMERIC])
    categorical = [column for column in clean if column not in NUMERIC]
    sampler = SMOTENC(
        categorical_features=categorical,
        sampling_strategy=ratio,
        random_state=seed,
        k_neighbors=5,
    )
    sampled_X, sampled_y = sampler.fit_resample(scaled, labels)
    synthetic = sampled_X.iloc[len(clean) :].copy()
    synthetic[NUMERIC] = scaler.inverse_transform(synthetic[NUMERIC])
    # Tenure is a discrete number of months. Round generated values before
    # validation; invalid structural combinations are never repaired.
    synthetic["tenure"] = synthetic["tenure"].round().astype(int)
    valid = _synthetic_valid(synthetic)
    accepted = synthetic.loc[valid].copy()
    if len(accepted):
        TelcoFeatureEngineer._validate(accepted)
    rejected = int((~valid).sum())
    all_X = pd.concat([clean, accepted], ignore_index=True)
    all_y = pd.concat(
        [labels, pd.Series(np.ones(len(accepted), dtype="int8"))], ignore_index=True
    )
    effective = float((all_y == 1).sum() / (all_y == 0).sum())
    return (
        all_X,
        all_y,
        {
            "requested_ratio": ratio,
            "effective_ratio": effective,
            "synthetic_generated": len(synthetic),
            "synthetic_rejected": rejected,
            "synthetic_accepted": len(accepted),
        },
    )


def fit_xgb_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    mode: str,
    parameters: dict,
    weight: str | float,
    smote_ratio: float | None,
    reduced: bool,
    seed: int,
    config_path: str,
    sampling_seed: int | None = None,
):
    """Fit a fresh pipeline; every learned step sees training rows only."""
    if smote_ratio is not None and weight != "unweighted":
        raise ValueError("SMOTE-NC cannot be combined with positive weighting.")
    sampling = {
        "effective_ratio": float((y == 1).sum() / (y == 0).sum()),
        "synthetic_generated": 0,
        "synthetic_rejected": 0,
        "synthetic_accepted": 0,
    }
    if smote_ratio is not None:
        X, y, sampling = resample_training(
            X, y, smote_ratio, sampling_seed if sampling_seed is not None else seed
        )
    if isinstance(weight, float):
        pos_weight = weight
    elif weight == "weighted":
        counts = y.value_counts()
        pos_weight = float(counts[0] / counts[1])
    else:
        pos_weight = None
    strategy = "weighted" if pos_weight is not None else "unweighted"
    stages = [("preprocess", make_preprocessor(mode, "tree", config_path=config_path))]
    if reduced:
        stages.append(
            (
                "selection",
                SelectFromModel(
                    make_model("xgboost", strategy, seed, parameters, pos_weight),
                    threshold="median",
                ),
            )
        )
    stages.append(
        ("estimator", make_model("xgboost", strategy, seed, parameters, pos_weight))
    )
    pipeline = Pipeline(stages)
    pipeline.fit(X, y)
    return pipeline, sampling, pos_weight


def evaluate_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    folds,
    config: dict,
    mode: str,
    parameters: dict,
    weight: str | float = "unweighted",
    smote_ratio: float | None = None,
    reduced: bool = False,
    config_path: str = "params.yaml",
) -> dict:
    """Use shared CV when possible; add fold-local resampling or selection."""
    if smote_ratio is not None and weight != "unweighted":
        raise ValueError("SMOTE-NC cannot be combined with positive weighting.")
    if not reduced and smote_ratio is None and weight in {"unweighted", "weighted"}:
        adjusted = copy.deepcopy(config)
        adjusted["model_training"]["models"]["xgboost"] = parameters
        return cross_validate_experiment(
            X, y, ("xgboost", mode, weight), adjusted, folds, config_path
        )
    seed = config["project"]["random_state"]
    probabilities = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    rows = []
    for number, (train_idx, valid_idx) in enumerate(folds, 1):
        train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
        started = time.perf_counter()
        pipeline, sampling, pos_weight = fit_xgb_pipeline(
            train_X,
            train_y,
            config,
            mode,
            parameters,
            weight,
            smote_ratio,
            reduced,
            seed,
            config_path,
            seed + number,
        )
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        fold_scores = positive_probabilities(pipeline, X.iloc[valid_idx])
        infer_seconds = time.perf_counter() - started
        probabilities[valid_idx] = fold_scores
        fold_ids[valid_idx] = number
        rows.append(
            {
                "fold": number,
                "training_seconds": train_seconds,
                "inference_seconds": infer_seconds,
                "scale_pos_weight": pos_weight,
                **sampling,
                **binary_metrics(y.iloc[valid_idx], fold_scores, 0.5),
            }
        )
    if (fold_ids < 0).any() or not np.isfinite(probabilities).all():
        raise ValueError("Incomplete XGBoost OOF predictions.")
    frame = pd.DataFrame(rows)
    return {
        "fold_metrics": frame,
        "oof_probabilities": probabilities,
        "fold_ids": fold_ids,
        "mean": {metric: float(frame[metric].mean()) for metric in SCORES},
        "std": {metric: float(frame[metric].std(ddof=1)) for metric in SCORES},
        "pooled": binary_metrics(y, probabilities, 0.5),
    }


def _save_result(output: Path, name: str, result: dict, y: pd.Series) -> None:
    result["fold_metrics"].to_csv(output / f"{name}_fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "fold": result["fold_ids"],
            "observed_churn": y.to_numpy(),
            "oof_probability": result["oof_probabilities"],
        }
    ).to_csv(output / f"{name}_oof.csv", index=False)
    _json(
        output / f"{name}_summary.json",
        {"mean": result["mean"], "std": result["std"], "pooled": result["pooled"]},
    )


def _load_result(output: Path, name: str) -> dict:
    summary = json.loads((output / f"{name}_summary.json").read_text())
    folds = pd.read_csv(output / f"{name}_fold_metrics.csv")
    oof = pd.read_csv(output / f"{name}_oof.csv")
    return {
        **summary,
        "fold_metrics": folds,
        "fold_ids": oof.fold.to_numpy(),
        "oof_probabilities": oof.oof_probability.to_numpy(),
    }


def _prior_result(output: Path, name: str, y: pd.Series) -> dict:
    """Read an existing Phase 6A result without fitting any model."""
    source = Path("reports/generated/model_optimization")
    board = pd.read_csv(source / "leaderboard.csv")
    row = board.loc[board.model.eq(name)].iloc[0]
    folds = pd.read_csv(source / f"{name}_fold_metrics.csv")
    oof = pd.read_csv(source / f"{name}_oof.csv")
    result = {
        "mean": {metric: float(row[f"mean_{metric}"]) for metric in SCORES},
        "std": {metric: float(row[f"std_{metric}"]) for metric in SCORES},
        "pooled": binary_metrics(y, oof.oof_probability, 0.5),
        "fold_metrics": folds,
        "fold_ids": oof.fold.to_numpy(),
        "oof_probabilities": oof.oof_probability.to_numpy(),
    }
    _save_result(output, f"prior_{name}", result, y)
    return result


def _screen(
    output: Path,
    name: str,
    X,
    y,
    folds,
    config,
    mode,
    parameters,
    weight="unweighted",
    smote_ratio=None,
    reduced=False,
    config_path="params.yaml",
) -> dict:
    if (output / f"{name}_summary.json").is_file():
        return _load_result(output, name)
    result = evaluate_xgboost(
        X,
        y,
        folds,
        config,
        mode,
        parameters,
        weight,
        smote_ratio,
        reduced,
        config_path,
    )
    _save_result(output, name, result, y)
    return result


def _suggest_extended(trial, oversampling: bool) -> tuple[dict, str | float]:
    parameters = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0001, 100, log=True),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "n_jobs": 1,
        "tree_method": "hist",
    }
    alpha_mode = trial.suggest_categorical("reg_alpha_mode", ["zero", "positive"])
    parameters["reg_alpha"] = (
        0.0
        if alpha_mode == "zero"
        else trial.suggest_float("reg_alpha", 0.0001, 10, log=True)
    )
    weight = (
        "unweighted"
        if oversampling
        else trial.suggest_float("scale_pos_weight", 1.0, 3.5)
    )
    return parameters, weight


def _old_scores(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "xgboost_initial":
        path = Path(
            "reports/generated/model_training/"
            "xgboost__baseline__weighted/oof_evaluation.csv"
        )
    else:
        model = "xgboost" if name == "xgboost_optimized" else name
        path = Path("reports/generated/model_optimization") / f"{model}_oof.csv"
    frame = pd.read_csv(path)
    return frame.observed_churn.to_numpy(), frame.oof_probability.to_numpy()


def _operating_rows(name: str, truth, scores, minimum_precision: float):
    table = operating_points(truth, scores)
    discrimination = binary_metrics(truth, scores, 0.5)
    default = table.loc[table.threshold.eq(0.5)].iloc[-1].to_dict()
    best_f1 = select_threshold(table)
    constrained = select_constrained_recall(table, minimum_precision)
    rows = []
    for policy, row in (
        ("default_0_5", default),
        ("maximum_f1", best_f1),
        ("recall_precision_floor", constrained),
    ):
        rows.append(
            {
                "model": name,
                "policy": policy,
                **(row or {"infeasible": True}),
                "average_precision": discrimination["average_precision"],
                "roc_auc": discrimination["roc_auc"],
            }
        )
    return rows, table


def run_enhancement(config_path="params.yaml") -> dict:
    """Screen, optimize, calibrate, and freeze a revised development choice."""
    config = load_config(config_path)
    validate_config(config)
    settings = config["xgboost_enhancement"]
    data_path = Path(config["data"]["development_path"])
    checksum = _checksum(data_path)
    prior = json.loads(
        Path(
            "reports/generated/model_optimization/selected_configuration.json"
        ).read_text()
    )
    if checksum != prior["dataset_sha256"]:
        raise ValueError("Development checksum differs from Phase 6A.")
    data = pd.read_parquet(data_path)
    X, y = split_features_target(data, config_path)
    seed = config["project"]["random_state"]
    folds = make_folds(y, 5, seed)
    fold_hash = _fold_hash(folds)
    if fold_hash != prior["fold_sha256"]:
        raise ValueError("Fold assignments differ from Phase 6A.")
    commit, dirty = _git_state()
    source_digest = hashlib.sha256()
    for source in (
        Path(__file__),
        Path("src/retention/features/engineering.py"),
        Path("src/retention/features/preprocessing.py"),
        Path("src/retention/evaluation/threshold.py"),
        Path("src/retention/models/optimization.py"),
    ):
        source_digest.update(source.read_bytes())
    source_sha256 = source_digest.hexdigest()
    output = Path(settings["report_directory"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "development_sha256": checksum,
        "fold_sha256": fold_hash,
        "seed": seed,
        "version": settings["search_space_version"],
        "prior_xgboost": json.loads(
            Path(
                "reports/generated/model_optimization/best_hyperparameters.json"
            ).read_text()
        )["xgboost"],
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Incompatible existing enhancement reports.")
    _json(manifest_path, manifest)
    mlflow.set_tracking_uri(config["model_training"]["mlflow_tracking_uri"])
    mlflow.set_experiment(settings["mlflow_experiment"])
    original_parameters = manifest["prior_xgboost"]["parameters"]
    baseline = _prior_result(output, "xgboost", y)
    feature_results = {"baseline": baseline}
    for mode in ("engineered", "enhanced"):
        feature_results[mode] = _screen(
            output,
            f"feature_{mode}",
            X,
            y,
            folds,
            config,
            mode,
            original_parameters,
            config_path=config_path,
        )
    feature_rows = [
        {
            "feature_mode": mode,
            "mean_ap": result["mean"]["average_precision"],
            "std_ap": result["std"]["average_precision"],
            "pooled_ap": result["pooled"]["average_precision"],
        }
        for mode, result in feature_results.items()
    ]
    pd.DataFrame(feature_rows).to_csv(output / "feature_screening.csv", index=False)
    existing_best = max(
        ("baseline", "engineered"),
        key=lambda m: feature_results[m]["mean"]["average_precision"],
    )
    mode = existing_best
    if (
        feature_results["enhanced"]["mean"]["average_precision"]
        > feature_results[existing_best]["mean"]["average_precision"]
        + settings["meaningful_ap_gain"]
    ):
        mode = "enhanced"
    unweighted = feature_results[mode]
    weighted = _screen(
        output,
        "weighted",
        X,
        y,
        folds,
        config,
        mode,
        original_parameters,
        weight="weighted",
        config_path=config_path,
    )
    balance_results = {"unweighted": unweighted, "weighted": weighted}
    for ratio in (0.5, 0.75, 1.0):
        key = f"smote_{ratio}"
        balance_results[key] = _screen(
            output,
            key,
            X,
            y,
            folds,
            config,
            mode,
            original_parameters,
            smote_ratio=ratio,
            config_path=config_path,
        )
    balance_rows = []
    for name, result in balance_results.items():
        folds_frame = result["fold_metrics"]
        balance_rows.append(
            {
                "method": name,
                "feature_mode": mode,
                "mean_ap": result["mean"]["average_precision"],
                "pooled_ap": result["pooled"]["average_precision"],
                "mean_recall_0_5": result["mean"]["recall"],
                "mean_precision_0_5": result["mean"]["precision"],
                "effective_ratio_mean": float(folds_frame["effective_ratio"].mean())
                if "effective_ratio" in folds_frame
                else None,
                "synthetic_rejected_total": int(folds_frame["synthetic_rejected"].sum())
                if "synthetic_rejected" in folds_frame
                else 0,
                "synthetic_accepted_total": int(folds_frame["synthetic_accepted"].sum())
                if "synthetic_accepted" in folds_frame
                else 0,
            }
        )
    pd.DataFrame(balance_rows).to_csv(output / "imbalance_screening.csv", index=False)
    balance_name = "unweighted"
    for name, result in balance_results.items():
        if (
            result["mean"]["average_precision"]
            > balance_results[balance_name]["mean"]["average_precision"]
            + settings["meaningful_ap_gain"]
        ):
            balance_name = name
    smote_ratio = (
        float(balance_name.removeprefix("smote_"))
        if balance_name.startswith("smote_")
        else None
    )
    weight = "weighted" if balance_name == "weighted" else "unweighted"
    full = balance_results[balance_name]
    reduced_result = _screen(
        output,
        "reduced",
        X,
        y,
        folds,
        config,
        mode,
        original_parameters,
        weight=weight,
        smote_ratio=smote_ratio,
        reduced=True,
        config_path=config_path,
    )
    reduced = bool(
        reduced_result["mean"]["average_precision"]
        > full["mean"]["average_precision"] + settings["meaningful_ap_gain"]
    )
    _json(
        output / "screening_decision.json",
        {
            "feature_mode": mode,
            "balance_method": balance_name,
            "smote_ratio": smote_ratio,
            "reduced_features": reduced,
            "reduced_ap": reduced_result["mean"]["average_precision"],
            "full_ap": full["mean"]["average_precision"],
            "meaningful_ap_gain": settings["meaningful_ap_gain"],
        },
    )
    storage_path = Path(settings["storage_path"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study_metadata = {
        **manifest,
        "feature_mode": mode,
        "smote_ratio": smote_ratio,
        "reduced": reduced,
        "screen_weight": weight,
        "search_space_version": settings["search_space_version"],
    }
    study = _study(
        "xgboost_enhancement",
        settings,
        study_metadata,
        f"sqlite:///{storage_path.resolve()}",
    )
    if not study.trials and mode == "baseline" and smote_ratio is None and not reduced:
        study.enqueue_trial(
            {
                "n_estimators": original_parameters["n_estimators"],
                "max_depth": original_parameters["max_depth"],
                "learning_rate": original_parameters["learning_rate"],
                "min_child_weight": original_parameters["min_child_weight"],
                "subsample": original_parameters["subsample"],
                "colsample_bytree": original_parameters["colsample_bytree"],
                "reg_lambda": original_parameters["reg_lambda"],
                "gamma": 0.0,
                "reg_alpha_mode": "zero",
                "scale_pos_weight": 1.0,
            }
        )
    complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = settings["trials"] - complete
    if remaining < 0:
        raise ValueError("Existing study exceeds the configured trial budget.")
    with mlflow.start_run(
        run_name="xgboost_extended_search", tags={"phase": "6A+"}
    ) as parent:
        mlflow.log_params(
            {
                "feature_mode": mode,
                "smote_ratio": str(smote_ratio),
                "reduced": reduced,
                "screen_weight": weight,
                "development_sha256": checksum,
                "fold_sha256": fold_hash,
                "seed": seed,
                "git_commit": commit,
                "git_dirty": dirty,
            }
        )
        for name in (
            "feature_screening.csv",
            "imbalance_screening.csv",
            "screening_decision.json",
        ):
            mlflow.log_artifact(str(output / name), artifact_path="screening")

        def objective(trial):
            parameters, trial_weight = _suggest_extended(trial, smote_ratio is not None)
            started = time.perf_counter()
            result = evaluate_xgboost(
                X,
                y,
                folds,
                config,
                mode,
                parameters,
                trial_weight,
                smote_ratio,
                reduced,
                config_path,
            )
            seconds = time.perf_counter() - started
            trial.set_user_attr("parameters", parameters)
            trial.set_user_attr("weight", trial_weight)
            trial.set_user_attr("seconds", seconds)
            trial.set_user_attr(
                "fold_ap", result["fold_metrics"].average_precision.tolist()
            )
            with mlflow.start_run(
                run_name=f"xgboost_trial_{trial.number}",
                nested=True,
                tags={
                    "optuna_trial": str(trial.number),
                    "parent_run_id": parent.info.run_id,
                },
            ):
                mlflow.log_params(
                    {
                        "feature_mode": mode,
                        "smote_ratio": str(smote_ratio),
                        "reduced": reduced,
                        "scale_pos_weight": str(trial_weight),
                        **{
                            f"hp_{key}": str(value) for key, value in parameters.items()
                        },
                    }
                )
                mlflow.log_metrics(
                    {
                        "mean_ap": result["mean"]["average_precision"],
                        "std_ap": result["std"]["average_precision"],
                        "pooled_ap": result["pooled"]["average_precision"],
                        "mean_recall_0_5": result["mean"]["recall"],
                        "mean_precision_0_5": result["mean"]["precision"],
                        "training_seconds": float(
                            result["fold_metrics"].training_seconds.sum()
                        ),
                        "inference_seconds": float(
                            result["fold_metrics"].inference_seconds.sum()
                        ),
                        **{
                            f"fold_{i}_ap": value
                            for i, value in enumerate(trial.user_attrs["fold_ap"], 1)
                        },
                    }
                )
            return result["mean"]["average_precision"]

        if remaining:
            study.optimize(objective, n_trials=remaining, n_jobs=1)
        if (
            sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
            != settings["trials"]
        ):
            raise RuntimeError("Extended search did not complete all 50 trials.")
        mlflow.log_metric("best_mean_ap", study.best_value)
    history = pd.DataFrame(
        [
            {
                "trial": trial.number,
                "state": trial.state.name,
                "mean_ap": trial.value,
                "seconds": trial.user_attrs.get("seconds"),
                "parameters": json.dumps(trial.params),
                "fold_ap": json.dumps(trial.user_attrs.get("fold_ap")),
            }
            for trial in study.trials
        ]
    )
    history.to_csv(output / "trial_history.csv", index=False)
    best = study.best_trial
    best_parameters = best.user_attrs["parameters"]
    best_weight = best.user_attrs["weight"]
    _json(
        output / "best_hyperparameters.json",
        {
            "trial": best.number,
            "feature_mode": mode,
            "parameters": best_parameters,
            "scale_pos_weight": best_weight,
            "smote_ratio": smote_ratio,
            "reduced": reduced,
            "mean_ap": best.value,
        },
    )
    best_name = f"extended_trial_{best.number}"
    extended = _screen(
        output,
        best_name,
        X,
        y,
        folds,
        config,
        mode,
        best_parameters,
        weight=best_weight,
        smote_ratio=smote_ratio,
        reduced=reduced,
        config_path=config_path,
    )
    retained_previous = bool(
        extended["mean"]["average_precision"] <= baseline["mean"]["average_precision"]
    )
    if retained_previous:
        chosen_result = baseline
        chosen_mode = "baseline"
        chosen_parameters = original_parameters
        chosen_weight = "unweighted"
        chosen_ratio = None
        chosen_reduced = False
    else:
        chosen_result = extended
        chosen_mode = mode
        chosen_parameters = best_parameters
        chosen_weight = best_weight
        chosen_ratio = smote_ratio
        chosen_reduced = reduced

    def fit_predict(
        X,
        y,
        train_idx,
        valid_idx,
        model,
        parameters,
        strategy,
        config,
        config_path,
        fold_number,
    ):
        pipeline, _, _ = fit_xgb_pipeline(
            X.iloc[train_idx],
            y.iloc[train_idx],
            config,
            chosen_mode,
            chosen_parameters,
            chosen_weight,
            chosen_ratio,
            chosen_reduced,
            seed,
            config_path,
            seed + fold_number,
        )
        return positive_probabilities(pipeline, X.iloc[valid_idx])

    calibration_path = output / "calibration_oof.csv"
    if calibration_path.exists():
        calibration_frame = pd.read_csv(calibration_path)
        raw = calibration_frame.raw_probability.to_numpy()
        calibrated = calibration_frame.sigmoid_probability.to_numpy()
    else:
        raw, calibrated = calibrated_oof(
            X,
            y,
            "xgboost",
            chosen_parameters,
            "unweighted",
            config,
            folds,
            config_path,
            fit_predict=fit_predict,
        )
        pd.DataFrame(
            {
                "observed_churn": y.to_numpy(),
                "raw_probability": raw,
                "sigmoid_probability": calibrated,
            }
        ).to_csv(calibration_path, index=False)
    calibration = pd.DataFrame(
        [
            {"strategy": "uncalibrated", **calibration_metrics(y, raw)},
            {"strategy": "sigmoid", **calibration_metrics(y, calibrated)},
        ]
    )
    calibration.to_csv(output / "calibration_comparison.csv", index=False)
    use_sigmoid = bool(
        calibration.iloc[1].brier < calibration.iloc[0].brier
        and calibration.iloc[1].log_loss < calibration.iloc[0].log_loss
    )
    enhanced_scores = calibrated if use_sigmoid else raw
    existing_board = pd.read_csv(
        "reports/generated/model_optimization/leaderboard.csv"
    ).set_index("model")
    initial = json.loads(
        Path(
            "reports/generated/model_training/"
            "xgboost__baseline__weighted/configuration.json"
        ).read_text()
    )
    candidate_meta = {
        "xgboost_initial": {
            "mean_ap": initial["mean"]["average_precision"],
            "std_ap": initial["std"]["average_precision"],
            "feature_mode": "baseline",
            "strategy": "weighted",
            "fit_seconds": 5 * initial["mean"]["training_seconds"],
            "inference_seconds": 5 * initial["mean"]["inference_seconds"],
        },
        "xgboost_optimized": {
            "mean_ap": float(existing_board.loc["xgboost", "mean_average_precision"]),
            "std_ap": float(existing_board.loc["xgboost", "std_average_precision"]),
            "feature_mode": "baseline",
            "strategy": "unweighted",
            "fit_seconds": float(
                existing_board.loc["xgboost", "total_training_seconds"]
            ),
            "inference_seconds": float(
                existing_board.loc["xgboost", "total_inference_seconds"]
            ),
        },
        "xgboost_enhanced": {
            "mean_ap": chosen_result["mean"]["average_precision"],
            "std_ap": chosen_result["std"]["average_precision"],
            "feature_mode": chosen_mode,
            "strategy": (
                "custom_scale_pos_weight"
                if isinstance(chosen_weight, float)
                else str(chosen_weight)
            ),
            "scale_pos_weight": chosen_weight,
            "fit_seconds": float(chosen_result["fold_metrics"].training_seconds.sum()),
            "inference_seconds": float(
                chosen_result["fold_metrics"].inference_seconds.sum()
            ),
        },
    }
    for family in ("logistic_regression", "random_forest", "ann"):
        row = existing_board.loc[family]
        candidate_meta[family] = {
            "mean_ap": float(row.mean_average_precision),
            "std_ap": float(row.std_average_precision),
            "feature_mode": str(row.feature_mode),
            "strategy": str(row.strategy),
            "fit_seconds": float(row.total_training_seconds),
            "inference_seconds": float(row.total_inference_seconds),
        }
    all_scores = {"xgboost_enhanced": enhanced_scores}
    for family in FAMILIES:
        truth, scores = _old_scores(family)
        if not np.array_equal(truth, y.to_numpy()):
            raise ValueError(f"Historical OOF labels differ for {family}.")
        all_scores[family] = scores
    threshold_rows = []
    leaderboard_rows = []
    operating = {}
    for family, scores in all_scores.items():
        rows, table = _operating_rows(
            family, y.to_numpy(), scores, settings["minimum_precision"]
        )
        threshold_rows.extend(rows)
        operating[family] = {row["policy"]: row for row in rows}
        meta = candidate_meta[family]
        pooled = binary_metrics(y, scores, 0.5)
        leaderboard_rows.append(
            {
                "model": family,
                **meta,
                **{f"pooled_{key}": value for key, value in pooled.items()},
                "constrained_recall": operating[family]["recall_precision_floor"].get(
                    "recall"
                ),
                "constrained_precision": operating[family][
                    "recall_precision_floor"
                ].get("precision"),
                "constrained_f1": operating[family]["recall_precision_floor"].get("f1"),
                "constrained_feasible": not operating[family][
                    "recall_precision_floor"
                ].get("infeasible", False),
            }
        )
        if family == "xgboost_enhanced":
            table.to_csv(output / "enhanced_threshold_curve.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(
        output / "threshold_comparison.csv", index=False
    )
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("mean_ap", ascending=False)
    leaderboard.to_csv(output / "consolidated_leaderboard.csv", index=False)
    paired = []
    chosen_folds = chosen_result["fold_metrics"].sort_values("fold")
    for family in (
        "xgboost_initial",
        "xgboost_optimized",
        "logistic_regression",
        "random_forest",
        "ann",
    ):
        if family == "xgboost_initial":
            path = Path(
                "reports/generated/model_training/xgboost__baseline__weighted/fold_metrics.csv"
            )
        else:
            model = "xgboost" if family == "xgboost_optimized" else family
            path = (
                Path("reports/generated/model_optimization")
                / f"{model}_fold_metrics.csv"
            )
        old_folds = pd.read_csv(path).sort_values("fold")
        for fold, delta in zip(
            chosen_folds.fold,
            chosen_folds.average_precision.to_numpy()
            - old_folds.average_precision.to_numpy(),
            strict=True,
        ):
            paired.append(
                {
                    "comparison": f"enhanced_minus_{family}",
                    "fold": int(fold),
                    "ap_difference": float(delta),
                }
            )
    pd.DataFrame(paired).to_csv(output / "paired_fold_differences.csv", index=False)
    leader_ap = float(leaderboard.iloc[0].mean_ap)
    eligible = leaderboard[
        (leaderboard.mean_ap >= leader_ap - 0.005) & leaderboard.constrained_feasible
    ].copy()
    if eligible.empty:
        selected = str(leaderboard.iloc[0].model)
        decision = "No candidate met the precision constraint; retained AP leader."
    else:
        maximum_recall = float(eligible.constrained_recall.max())
        competitive = eligible[eligible.constrained_recall >= maximum_recall - 0.02]
        simplicity = {
            "logistic_regression": 0,
            "random_forest": 1,
            "xgboost_initial": 2,
            "xgboost_optimized": 2,
            "xgboost_enhanced": 2,
            "ann": 3,
        }
        selected = min(
            competitive.model,
            key=lambda name: (
                simplicity[name],
                -candidate_meta[name]["mean_ap"],
            ),
        )
        decision = (
            "Mean fold AP primary; within 0.005 of the leader, keep models "
            "within 0.02 constrained recall of the best, then prefer lower "
            "complexity. Both margins are engineering heuristics."
        )
    selected_point = operating[selected]["recall_precision_floor"]
    if selected_point.get("infeasible"):
        selected_point = operating[selected]["maximum_f1"]
        decision += " Precision floor infeasible; maximum F1 fallback."
    previous_best = json.loads(
        Path(
            "reports/generated/model_optimization/best_hyperparameters.json"
        ).read_text()
    )
    if selected == "xgboost_enhanced":
        selected_parameters = chosen_parameters
        selected_ratio = chosen_ratio
        selected_reduced = chosen_reduced
        selected_calibration = "sigmoid" if use_sigmoid else "uncalibrated"
    elif selected == "xgboost_initial":
        selected_parameters = initial["hyperparameters"]
        selected_ratio = None
        selected_reduced = False
        selected_calibration = "uncalibrated"
    else:
        model_key = "xgboost" if selected == "xgboost_optimized" else selected
        selected_parameters = previous_best[model_key]["parameters"]
        selected_ratio = None
        selected_reduced = False
        selected_calibration = "uncalibrated"
    default_point = operating[selected]["default_0_5"]
    selected_config = {
        "model": selected,
        "feature_mode": candidate_meta[selected]["feature_mode"],
        "strategy": candidate_meta[selected]["strategy"],
        "scale_pos_weight": candidate_meta[selected].get("scale_pos_weight"),
        "parameters": selected_parameters,
        "smote_ratio": selected_ratio,
        "reduced_features": selected_reduced,
        "calibration": selected_calibration,
        "threshold": selected_point["threshold"],
        "minimum_precision": settings["minimum_precision"],
        "default_metrics": default_point,
        "selected_metrics": selected_point,
        "additional_churners_identified": selected_point["tp"] - default_point["tp"],
        "additional_false_alerts": selected_point["fp"] - default_point["fp"],
        "reduction_in_missed_churners": default_point["fn"] - selected_point["fn"],
        "decision": decision,
        "xgboost_retained_previous": retained_previous,
        "development_sha256": checksum,
        "fold_sha256": fold_hash,
        "seed": seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": source_sha256,
        "limitations": (
            "Repeated development-fold selection may be optimistic; "
            "no held-out evaluation or final refit."
        ),
    }
    _json(output / "revised_selection.json", selected_config)
    fig, ax = plt.subplots(figsize=(8, 4))
    display = leaderboard[
        leaderboard.model.isin(
            [
                "xgboost_initial",
                "xgboost_optimized",
                "xgboost_enhanced",
                "logistic_regression",
            ]
        )
    ]
    ax.bar(display.model, display.mean_ap, color="tab:blue")
    ax.set(ylabel="Mean five-fold AP", title="Development model comparison")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "before_after.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for family in (
        "xgboost_initial",
        "xgboost_optimized",
        "xgboost_enhanced",
        "logistic_regression",
    ):
        precision, recall, _ = precision_recall_curve(y, all_scores[family])
        ax.plot(recall, precision, label=family)
    ax.axhline(y.mean(), color="gray", linestyle="--", label="prevalence")
    ax.set(xlabel="Recall", ylabel="Precision")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "precision_recall.png", dpi=130)
    plt.close(fig)
    curve = pd.read_csv(output / "enhanced_threshold_curve.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ("precision", "recall", "f1"):
        ax.plot(curve.threshold, curve[metric], label=metric)
    ax.axhline(settings["minimum_precision"], color="gray", linestyle="--")
    ax.set(xlabel="Threshold", ylabel="Pooled OOF score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "threshold_metrics.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    for name, scores in (("Uncalibrated", raw), ("Sigmoid", calibrated)):
        observed, predicted = reliability_points(y, scores)
        ax.plot(predicted, observed, marker="o", label=name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "calibration_reliability.png", dpi=130)
    plt.close(fig)
    with mlflow.start_run(
        run_name="xgboost_enhancement_selection", tags={"phase": "6A+"}
    ) as run:
        selected_config["mlflow_run_id"] = run.info.run_id
        _json(output / "revised_selection.json", selected_config)
        mlflow.log_params(
            {
                "selected_model": selected,
                "selected_threshold": selected_point["threshold"],
                "minimum_precision": settings["minimum_precision"],
                "development_sha256": checksum,
                "fold_sha256": fold_hash,
                "git_commit": commit,
                "source_sha256": source_sha256,
            }
        )
        mlflow.log_metrics(
            {
                "selected_mean_ap": candidate_meta[selected]["mean_ap"],
                "selected_constrained_recall": selected_point["recall"],
                "selected_constrained_precision": selected_point["precision"],
                "extended_best_mean_ap": best.value,
            }
        )
        mlflow.log_artifacts(str(output))
    return selected_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    print(json.dumps(run_enhancement(args.config), indent=2, default=float))


if __name__ == "__main__":
    main()
