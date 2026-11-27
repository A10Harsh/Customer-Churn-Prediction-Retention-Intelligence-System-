"""Reproducible development-only Phase 6A optimization and selection."""

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
from sklearn.model_selection import train_test_split

from retention.evaluation.calibration import (
    apply_sigmoid,
    calibration_metrics,
    fit_sigmoid,
    reliability_points,
)
from retention.evaluation.metrics import binary_metrics, positive_probabilities
from retention.evaluation.threshold import select_threshold, threshold_table
from retention.features.preprocessing import make_preprocessor, split_features_target
from retention.models.ann import predict_probabilities, resolve_device, train_network
from retention.models.ann_training import cross_validate_ann
from retention.models.factory import make_estimator_pipeline
from retention.models.training import (
    _checksum,
    _git_state,
    cross_validate_experiment,
    make_folds,
)
from retention.utils.config import load_config

MODELS = ("logistic_regression", "random_forest", "xgboost", "ann")
MODES = {
    "logistic_regression": "engineered",
    "random_forest": "engineered",
    "xgboost": "baseline",
    "ann": "baseline",
}


def validate_config(config: dict) -> None:
    """Reject unsupported or incomplete optimization settings."""
    settings = config["model_optimization"]
    if config["validation"]["cv_folds"] != 5 or not isinstance(
        config["project"]["random_state"], int
    ):
        raise ValueError("Phase 6A requires five folds and an integer seed.")
    if settings["search_space_version"] != 1 or set(settings["trial_budgets"]) != set(
        MODELS
    ):
        raise ValueError("Unsupported search-space version or model budgets.")
    if any(not isinstance(n, int) or n < 1 for n in settings["trial_budgets"].values()):
        raise ValueError("Trial budgets must be positive integers.")
    development = Path(config["data"]["development_path"])
    test = Path(config["data"]["test_path"])
    if (
        development.name != "development.parquet"
        or development.resolve() == test.resolve()
    ):
        raise ValueError("Only the configured development.parquet may be used.")


def suggest_parameters(trial: optuna.Trial, model: str) -> tuple[dict, str]:
    """Return requested search space and imbalance strategy."""
    strategy = trial.suggest_categorical("strategy", ["unweighted", "weighted"])
    if model == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.001, 100, log=True),
            "max_iter": 2000,
            "solver": "lbfgs",
        }, strategy
    if model == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_categorical(
                "max_depth", [3, 5, 8, 12, 20, None]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "n_jobs": 2,
        }, strategy
    if model == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10, log=True),
            "n_jobs": 1,
            "tree_method": "hist",
        }, strategy
    if model == "ann":
        return {
            "hidden_dims": list(
                trial.suggest_categorical(
                    "hidden_dims", ["32,16", "64,32", "64,32,16"]
                ).split(",")
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.0001, 0.005, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", 0.000001, 0.01, log=True
            ),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }, strategy
    raise ValueError(f"Unknown model: {model}")


def _ann_settings(config: dict, parameters: dict) -> dict:
    settings = copy.deepcopy(config)
    settings["ann_training"].update(parameters)
    settings["ann_training"]["hidden_dims"] = [
        int(n) for n in parameters["hidden_dims"]
    ]
    return settings


def evaluate_candidate(X, y, model, parameters, strategy, config, folds, config_path):
    """Reuse existing classical and ANN cross-validation routines."""
    if model == "ann":
        settings = _ann_settings(config, parameters)
        device = resolve_device(settings["ann_training"]["device"])
        return cross_validate_ann(
            X, y, MODES[model], strategy, settings, folds, device, config_path
        )
    settings = copy.deepcopy(config)
    settings["model_training"]["models"][model] = parameters
    return cross_validate_experiment(
        X, y, (model, MODES[model], strategy), settings, folds, config_path
    )


def _fold_hash(folds) -> str:
    digest = hashlib.sha256()
    for train, valid in folds:
        digest.update(np.asarray(train, dtype="<i8").tobytes())
        digest.update(np.asarray(valid, dtype="<i8").tobytes())
    return digest.hexdigest()


def _study(model, settings, metadata, storage):
    name = f"phase6a_{model}"
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=metadata["seed"]),
    )
    if study.user_attrs:
        if study.user_attrs != metadata:
            raise ValueError(f"Incompatible existing study: {name}")
    elif study.trials:
        raise ValueError(f"Existing study has no compatibility metadata: {name}")
    else:
        for key, value in metadata.items():
            study.set_user_attr(key, value)
    return study


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=float))


def _fit_predict(
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
    """Fit one fresh model using training indices only."""
    seed = config["project"]["random_state"]
    if model == "ann":
        settings = _ann_settings(config, parameters)["ann_training"]
        device = resolve_device(settings["device"])
        inner_train, inner_valid = train_test_split(
            train_idx,
            test_size=settings["inner_validation_size"],
            stratify=y.iloc[train_idx],
            random_state=seed + fold_number,
        )
        prep = make_preprocessor(MODES[model], "neural", config_path=config_path)
        inner_X = prep.fit_transform(X.iloc[inner_train])
        valid_X = prep.transform(X.iloc[inner_valid])
        _, _, epoch, _ = train_network(
            inner_X,
            y.iloc[inner_train].to_numpy(),
            settings,
            device,
            seed + fold_number,
            strategy == "weighted",
            settings["max_epochs"],
            valid_X,
            y.iloc[inner_valid].to_numpy(),
        )
        prep = make_preprocessor(MODES[model], "neural", config_path=config_path)
        full_X = prep.fit_transform(X.iloc[train_idx])
        network, _, _, _ = train_network(
            full_X,
            y.iloc[train_idx].to_numpy(),
            settings,
            device,
            seed + fold_number,
            strategy == "weighted",
            epoch,
        )
        return predict_probabilities(network, prep.transform(X.iloc[valid_idx]), device)
    counts = y.iloc[train_idx].value_counts()
    weight = (
        float(counts[0] / counts[1])
        if model == "xgboost" and strategy == "weighted"
        else None
    )
    pipeline = make_estimator_pipeline(
        model, MODES[model], strategy, seed, parameters, weight, config_path
    )
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    return positive_probabilities(pipeline, X.iloc[valid_idx])


def calibrated_oof(
    X, y, model, parameters, strategy, config, folds, config_path, fit_predict=None
):
    """Fit an inner OOF calibrator independently inside each outer fold."""
    calibrated = np.full(len(y), np.nan)
    raw = np.full(len(y), np.nan)
    seed = config["project"]["random_state"]
    fit_function = fit_predict or _fit_predict
    for number, (outer_train, outer_valid) in enumerate(folds, 1):
        inner_y = y.iloc[outer_train].reset_index(drop=True)
        inner_folds = make_folds(inner_y, 5, seed)
        inner_scores = np.full(len(outer_train), np.nan)
        for inner_number, (inner_train, inner_valid) in enumerate(inner_folds, 1):
            scores = fit_function(
                X,
                y,
                outer_train[inner_train],
                outer_train[inner_valid],
                model,
                parameters,
                strategy,
                config,
                config_path,
                number * 10 + inner_number,
            )
            inner_scores[inner_valid] = scores
        if not np.isfinite(inner_scores).all():
            raise ValueError("Incomplete inner OOF calibration predictions.")
        calibrator = fit_sigmoid(inner_scores, y.iloc[outer_train])
        raw[outer_valid] = fit_function(
            X,
            y,
            outer_train,
            outer_valid,
            model,
            parameters,
            strategy,
            config,
            config_path,
            number,
        )
        calibrated[outer_valid] = apply_sigmoid(calibrator, raw[outer_valid])
    return raw, calibrated


def run_optimization(config_path="params.yaml"):
    """Execute or resume 70 trials and write Phase 6A selection artifacts."""
    config = load_config(config_path)
    validate_config(config)
    settings = config["model_optimization"]
    data_path = Path(config["data"]["development_path"])
    data = pd.read_parquet(data_path)
    X, y = split_features_target(data, config_path)
    seed = config["project"]["random_state"]
    folds = make_folds(y, 5, seed)
    fold_hash = _fold_hash(folds)
    checksum = _checksum(data_path)
    commit, dirty = _git_state()
    output = Path(settings["report_directory"])
    output.mkdir(parents=True, exist_ok=True)
    storage_path = Path(settings["storage_path"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path.resolve()}"
    mlflow.set_tracking_uri(config["model_training"]["mlflow_tracking_uri"])
    mlflow.set_experiment(settings["mlflow_experiment"])
    results = {}
    histories = []
    for model in MODELS:
        metadata = {
            "dataset_sha256": checksum,
            "fold_sha256": fold_hash,
            "seed": seed,
            "search_space_version": settings["search_space_version"],
            "feature_mode": MODES[model],
            "base_config": config["ann_training"]
            if model == "ann"
            else config["model_training"]["models"][model],
        }
        study = _study(model, settings, metadata, storage)
        completed = sum(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
        )
        remaining = settings["trial_budgets"][model] - completed
        if remaining < 0:
            raise ValueError(f"Study {model} exceeds configured trial budget.")
        with mlflow.start_run(
            run_name=model, tags={"phase": "6A", "model": model}
        ) as parent:
            mlflow.log_params(
                {
                    "model": model,
                    "feature_mode": MODES[model],
                    "development_sha256": checksum,
                    "fold_sha256": fold_hash,
                    "seed": seed,
                    "git_commit": commit,
                    "git_dirty": dirty,
                }
            )

            def objective(trial, model=model):
                parameters, strategy = suggest_parameters(trial, model)
                started = time.perf_counter()
                result = evaluate_candidate(
                    X, y, model, parameters, strategy, config, folds, config_path
                )
                elapsed = time.perf_counter() - started
                trial.set_user_attr(
                    "fold_ap", result["fold_metrics"]["average_precision"].tolist()
                )
                trial.set_user_attr("seconds", elapsed)
                trial.set_user_attr("parameters", parameters)
                with mlflow.start_run(
                    run_name=f"{model}_trial_{trial.number}",
                    nested=True,
                    tags={
                        "optuna_trial": str(trial.number),
                        "parent_run_id": parent.info.run_id,
                    },
                ):
                    mlflow.log_params(
                        {
                            "model": model,
                            "strategy": strategy,
                            "feature_mode": MODES[model],
                            **{f"hp_{k}": str(v) for k, v in parameters.items()},
                        }
                    )
                    mlflow.log_metrics(
                        {
                            "mean_ap": result["mean"]["average_precision"],
                            "std_ap": result["std"]["average_precision"],
                            "training_seconds": result["fold_metrics"][
                                "training_seconds"
                            ].sum(),
                            "inference_seconds": result["fold_metrics"][
                                "inference_seconds"
                            ].sum(),
                            **{
                                f"fold_{i}_ap": ap
                                for i, ap in enumerate(trial.user_attrs["fold_ap"], 1)
                            },
                        }
                    )
                return result["mean"]["average_precision"]

            if remaining:
                study.optimize(objective, n_trials=remaining, n_jobs=1)
            if (
                sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
                != settings["trial_budgets"][model]
            ):
                raise RuntimeError(f"Incomplete optimization study: {model}")
            best = study.best_trial
            parameters = dict(best.user_attrs["parameters"])
            if model == "logistic_regression":
                parameters.pop("penalty", None)  # sklearn defaults to L2.
            if model == "ann":
                parameters["hidden_dims"] = [int(n) for n in parameters["hidden_dims"]]
            strategy = best.params["strategy"]
            result = evaluate_candidate(
                X, y, model, parameters, strategy, config, folds, config_path
            )
            results[model] = {
                "parameters": parameters,
                "strategy": strategy,
                "result": result,
                "best_trial": best.number,
            }
            mlflow.log_metrics(
                {
                    "best_mean_ap": result["mean"]["average_precision"],
                    "best_std_ap": result["std"]["average_precision"],
                }
            )
            for trial in study.trials:
                histories.append(
                    {
                        "model": model,
                        "trial": trial.number,
                        "state": trial.state.name,
                        "mean_ap": trial.value,
                        "seconds": trial.user_attrs.get("seconds"),
                        "parameters": json.dumps(trial.params),
                        "fold_ap": json.dumps(trial.user_attrs.get("fold_ap")),
                    }
                )
        print(f"{model}: best AP {best.value:.6f}", flush=True)
    pd.DataFrame(histories).to_csv(output / "trial_history.csv", index=False)
    _json(
        output / "best_hyperparameters.json",
        {
            model: {
                "strategy": value["strategy"],
                "feature_mode": MODES[model],
                "parameters": value["parameters"],
                "trial": value["best_trial"],
            }
            for model, value in results.items()
        },
    )
    # A search cannot displace a stronger initial configuration on the same folds.
    for model, value in results.items():
        source = (
            Path("reports/generated/ann_training")
            if model == "ann"
            else Path("reports/generated/model_training")
        )
        initial = pd.read_csv(source / "leaderboard.csv")
        initial = initial[initial.model == model].sort_values(
            "ap_mean", ascending=False
        )
        initial_ap = float(initial.iloc[0].ap_mean)
        value["search_mean_ap"] = value["result"]["mean"]["average_precision"]
        if initial_ap > value["search_mean_ap"]:
            initial_strategy = str(initial.iloc[0].strategy)
            initial_parameters = (
                {
                    key: config["ann_training"][key]
                    for key in (
                        "hidden_dims",
                        "dropout",
                        "learning_rate",
                        "weight_decay",
                        "batch_size",
                    )
                }
                if model == "ann"
                else config["model_training"]["models"][model]
            )
            if model == "xgboost":
                initial_parameters = {**initial_parameters, "n_jobs": 1}
            value.update(
                parameters=initial_parameters,
                strategy=initial_strategy,
                retained_initial=True,
                result=evaluate_candidate(
                    X,
                    y,
                    model,
                    initial_parameters,
                    initial_strategy,
                    config,
                    folds,
                    config_path,
                ),
            )
        else:
            value["retained_initial"] = False
    leaderboard = []
    for model, value in results.items():
        result = value["result"]
        leaderboard.append(
            {
                "model": model,
                "feature_mode": MODES[model],
                "strategy": value["strategy"],
                "retained_initial": value["retained_initial"],
                **{f"mean_{k}": v for k, v in result["mean"].items()},
                **{f"std_{k}": v for k, v in result["std"].items()},
                **{f"pooled_{k}": v for k, v in result["pooled"].items()},
                "total_training_seconds": float(
                    result["fold_metrics"]["training_seconds"].sum()
                ),
                "total_inference_seconds": float(
                    result["fold_metrics"]["inference_seconds"].sum()
                ),
            }
        )
        result["fold_metrics"].to_csv(output / f"{model}_fold_metrics.csv", index=False)
        pd.DataFrame(
            {
                "fold": result["fold_ids"],
                "observed_churn": y.to_numpy(),
                "oof_probability": result["oof_probabilities"],
            }
        ).to_csv(output / f"{model}_oof.csv", index=False)
    board = pd.DataFrame(leaderboard).sort_values(
        "mean_average_precision", ascending=False
    )
    board.to_csv(output / "leaderboard.csv", index=False)
    previous = []
    for model in MODELS:
        path = (
            Path(
                "reports/generated/ann_training"
                if model == "ann"
                else "reports/generated/model_training"
            )
            / "leaderboard.csv"
        )
        old = pd.read_csv(path)
        old = old[old["model"] == model]
        previous.append(
            {
                "model": model,
                "initial_mean_ap": float(old["ap_mean"].max()),
                "optimized_mean_ap": results[model]["result"]["mean"][
                    "average_precision"
                ],
            }
        )
    comparison = pd.DataFrame(previous)
    comparison.to_csv(output / "before_after.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(MODELS))
    ax.bar(positions - 0.2, comparison.initial_mean_ap, 0.4, label="Initial")
    ax.bar(positions + 0.2, comparison.optimized_mean_ap, 0.4, label="Optimized")
    ax.set_xticks(positions, MODELS, rotation=15)
    ax.set_ylabel("Mean five-fold AP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "before_after.png", dpi=130)
    plt.close(fig)
    leader_ap = float(board.iloc[0]["mean_average_precision"])
    eligible = board[board.mean_average_precision >= leader_ap - 0.005]
    simplicity = {"logistic_regression": 0, "random_forest": 1, "xgboost": 2, "ann": 3}
    selected = min(eligible.model, key=simplicity.get)
    selected_value = results[selected]
    raw, calibrated = calibrated_oof(
        X,
        y,
        selected,
        selected_value["parameters"],
        selected_value["strategy"],
        config,
        folds,
        config_path,
    )
    calibration = pd.DataFrame(
        [
            {"strategy": name, **calibration_metrics(y, scores)}
            for name, scores in (("uncalibrated", raw), ("sigmoid", calibrated))
        ]
    )
    calibration.to_csv(output / "calibration_comparison.csv", index=False)
    use_sigmoid = bool(
        calibration.iloc[1].brier < calibration.iloc[0].brier
        and calibration.iloc[1].log_loss < calibration.iloc[0].log_loss
    )
    selected_scores = calibrated if use_sigmoid else raw
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    for name, scores in (("Uncalibrated", raw), ("Sigmoid", calibrated)):
        observed, predicted = reliability_points(y, scores)
        ax.plot(predicted, observed, marker="o", label=name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "calibration_reliability.png", dpi=130)
    plt.close(fig)
    table = threshold_table(y, selected_scores)
    table.to_csv(output / "threshold_evaluation.csv", index=False)
    optimal = select_threshold(table)
    default = binary_metrics(y, selected_scores, 0.5)
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ("precision", "recall", "f1"):
        ax.plot(table.threshold, table[metric], label=metric)
    ax.axvline(optimal["threshold"], color="gray", linestyle="--")
    ax.set(xlabel="Classification threshold", ylabel="Pooled OOF score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "threshold_metrics.png", dpi=130)
    plt.close(fig)
    frozen = {
        "model": selected,
        "feature_mode": MODES[selected],
        "strategy": selected_value["strategy"],
        "parameters": selected_value["parameters"],
        "calibration": "sigmoid" if use_sigmoid else "uncalibrated",
        "threshold": optimal["threshold"],
        "threshold_0_5_metrics": default,
        "selected_threshold_metrics": optimal,
        "selection_note": (
            "Within 0.005 mean AP of leader, prefer simpler model; "
            "engineering heuristic, not statistical equivalence."
        ),
        "dataset_sha256": checksum,
        "fold_sha256": fold_hash,
        "seed": seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "limitations": (
            "Selection-stage estimates reuse CV folds; "
            "no held-out evaluation or production refit."
        ),
    }
    _json(output / "selected_configuration.json", frozen)
    with mlflow.start_run(run_name="phase6a_selection", tags={"phase": "6A"}):
        mlflow.log_params(
            {
                "selected_model": selected,
                "calibration": frozen["calibration"],
                "threshold": optimal["threshold"],
                "development_sha256": checksum,
                "fold_sha256": fold_hash,
            }
        )
        mlflow.log_metrics(
            {
                "selected_mean_ap": selected_value["result"]["mean"][
                    "average_precision"
                ],
                "selected_pooled_f1": optimal["f1"],
            }
        )
        mlflow.log_artifacts(str(output))
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    print(json.dumps(run_optimization(args.config), indent=2, default=float))


if __name__ == "__main__":
    main()
