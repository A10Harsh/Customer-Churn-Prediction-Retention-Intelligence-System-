"""Leakage-safe nested ANN fitting on the development split only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from retention.evaluation.metrics import binary_metrics
from retention.features.preprocessing import make_preprocessor, split_features_target
from retention.models.ann import (
    predict_probabilities,
    resolve_device,
    train_network,
)
from retention.models.training import SCORES, _checksum, _git_state, make_folds
from retention.utils.config import load_config
from retention.utils.logger import get_logger

LOGGER = get_logger(__name__)


def cross_validate_ann(
    X: pd.DataFrame,
    y: pd.Series,
    mode: str,
    strategy: str,
    config: dict,
    folds: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    config_path: str = "params.yaml",
) -> dict:
    """Select epochs on an inner split; refit before each untouched outer fold."""
    if mode not in {"baseline", "engineered"} or strategy not in {
        "unweighted",
        "weighted",
    }:
        raise ValueError("Unknown ANN experiment configuration.")
    settings = config["ann_training"]
    seed = config["project"]["random_state"]
    threshold = config["model_training"]["threshold"]
    probabilities = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    metrics_rows, history_rows = [], []
    dimensions, parameters = [], []
    for fold, (outer_train, outer_valid) in enumerate(folds, start=1):
        if (fold_ids[outer_valid] != -1).any():
            raise ValueError("Outer validation folds overlap.")
        inner_train, inner_valid = train_test_split(
            outer_train,
            test_size=settings["inner_validation_size"],
            stratify=y.iloc[outer_train],
            random_state=seed + fold,
        )
        if y.iloc[inner_valid].nunique() != 2:
            raise ValueError("Inner validation split must contain both classes.")
        if (
            np.intersect1d(inner_train, outer_valid).size
            or np.intersect1d(inner_valid, outer_valid).size
        ):
            raise ValueError("Outer validation rows entered inner selection.")
        started = time.perf_counter()
        inner_preprocessor = make_preprocessor(mode, "neural", config_path=config_path)
        inner_X = inner_preprocessor.fit_transform(X.iloc[inner_train])
        inner_validation_X = inner_preprocessor.transform(X.iloc[inner_valid])
        _, inner_history, best_epoch, inner_weight = train_network(
            inner_X,
            y.iloc[inner_train].to_numpy(),
            settings,
            device,
            seed + fold,
            strategy == "weighted",
            settings["max_epochs"],
            inner_validation_X,
            y.iloc[inner_valid].to_numpy(),
        )
        for row in inner_history:
            history_rows.append({"fold": fold, "phase": "selection", **row})

        # Fresh preprocessing and network; selected epoch count is the only
        # information carried from the inner fit into the outer refit.
        outer_preprocessor = make_preprocessor(mode, "neural", config_path=config_path)
        outer_X = outer_preprocessor.fit_transform(X.iloc[outer_train])
        model, refit_history, _, refit_weight = train_network(
            outer_X,
            y.iloc[outer_train].to_numpy(),
            settings,
            device,
            seed + fold,
            strategy == "weighted",
            best_epoch,
        )
        training_seconds = time.perf_counter() - started
        for row in refit_history:
            history_rows.append({"fold": fold, "phase": "refit", **row})
        dimensions.append(outer_X.shape[1])
        parameters.append(sum(parameter.numel() for parameter in model.parameters()))

        started = time.perf_counter()
        outer_validation_X = outer_preprocessor.transform(X.iloc[outer_valid])
        fold_probability = predict_probabilities(model, outer_validation_X, device)
        inference_seconds = time.perf_counter() - started
        probabilities[outer_valid] = fold_probability
        fold_ids[outer_valid] = fold
        metrics_rows.append(
            {
                "fold": fold,
                "outer_train_rows": len(outer_train),
                "inner_train_rows": len(inner_train),
                "inner_validation_rows": len(inner_valid),
                "outer_validation_rows": len(outer_valid),
                "inner_pos_weight": inner_weight,
                "refit_pos_weight": refit_weight,
                "best_epoch": best_epoch,
                "selection_epochs": len(inner_history),
                "input_dimension": outer_X.shape[1],
                "parameter_count": parameters[-1],
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                **binary_metrics(y.iloc[outer_valid], fold_probability, threshold),
            }
        )
        LOGGER.info(
            "ANN %s %s fold %s: best epoch %s", mode, strategy, fold, best_epoch
        )
    if (fold_ids < 0).any() or not np.isfinite(probabilities).all():
        raise ValueError("Every development row needs exactly one ANN OOF probability.")
    fold_frame = pd.DataFrame(metrics_rows)
    return {
        "fold_metrics": fold_frame,
        "history": pd.DataFrame(history_rows),
        "oof_probabilities": probabilities,
        "fold_ids": fold_ids,
        "pooled": binary_metrics(y, probabilities, threshold),
        "mean": {name: float(fold_frame[name].mean()) for name in SCORES},
        "std": {name: float(fold_frame[name].std(ddof=1)) for name in SCORES},
        "input_dimensions": dimensions,
        "parameter_counts": parameters,
    }


def _plot_history(result: dict, output: Path) -> Path:
    history = result["history"]
    fold = history[(history.fold == 1) & (history.phase == "selection")]
    fig, left = plt.subplots(figsize=(7, 4))
    left.plot(fold.epoch, fold.train_loss, label="Training loss")
    left.plot(fold.epoch, fold.valid_loss, label="Inner validation loss")
    left.set(xlabel="Epoch", ylabel="BCE loss")
    right = left.twinx()
    right.plot(
        fold.epoch, fold.valid_ap, color="tab:green", label="Inner validation AP"
    )
    right.set_ylabel("Average Precision")
    left.axvline(
        int(result["fold_metrics"].iloc[0].best_epoch), color="gray", linestyle="--"
    )
    left.legend(loc="upper left", fontsize=8)
    right.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = output / "representative_training_history.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_comparison(
    result: dict,
    y: pd.Series,
    classical: pd.DataFrame,
    classical_root: Path,
    output: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    precision, recall, _ = precision_recall_curve(y, result["oof_probabilities"])
    ax.plot(
        recall, precision, label=f"ANN (AP={result['pooled']['average_precision']:.3f})"
    )
    for model in ("logistic_regression", "random_forest", "xgboost"):
        winner = (
            classical[classical.model == model]
            .sort_values("ap_mean", ascending=False)
            .iloc[0]
        )
        table = pd.read_csv(classical_root / winner["name"] / "oof_evaluation.csv")
        if not np.array_equal(table.observed_churn.to_numpy(), y.to_numpy()):
            raise ValueError("Classical and ANN OOF labels do not align.")
        precision, recall, _ = precision_recall_curve(y, table.oof_probability)
        pooled_ap = binary_metrics(y, table.oof_probability)["average_precision"]
        ax.plot(recall, precision, label=f"{model} (OOF AP={pooled_ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output / "ann_classical_precision_recall.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_ann_experiments(config_path: str = "params.yaml") -> pd.DataFrame:
    """Run four ANN configurations and log local MLflow and report artifacts."""
    config = load_config(config_path)
    data_path = Path(config["data"]["development_path"])
    if data_path.resolve() == Path(config["data"]["test_path"]).resolve():
        raise ValueError("Development and held-out paths must differ.")
    classical_root = Path(config["model_training"]["report_directory"])
    classical = pd.read_csv(classical_root / "leaderboard.csv")
    data = pd.read_parquet(data_path)
    X, y = split_features_target(data, config_path)
    settings = config["ann_training"]
    seed = config["project"]["random_state"]
    torch.set_num_threads(1)
    device = resolve_device(settings["device"])
    folds = make_folds(y, config["validation"]["cv_folds"], seed)
    fold_digest = hashlib.sha256(
        b"".join(valid.astype(np.int64).tobytes() for _, valid in folds)
    ).hexdigest()
    checksum, (commit, dirty) = _checksum(data_path), _git_state()
    report = Path(settings["report_directory"])
    report.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config["model_training"]["mlflow_tracking_uri"])
    mlflow.set_experiment(settings["mlflow_experiment"])
    results = []
    for mode in settings["feature_modes"]:
        for strategy in settings["loss_strategies"]:
            name = f"ann__{mode}__{strategy}"
            LOGGER.info("Training %s on %s", name, device)
            result = cross_validate_ann(
                X, y, mode, strategy, config, folds, device, config_path
            )
            result.update(name=name, feature_mode=mode, strategy=strategy)
            folder = report / name
            folder.mkdir(exist_ok=True)
            result["fold_metrics"].to_csv(folder / "fold_metrics.csv", index=False)
            result["history"].to_csv(folder / "training_history.csv", index=False)
            pd.DataFrame(
                {
                    "fold": result["fold_ids"],
                    "observed_churn": y.to_numpy(),
                    "oof_probability": result["oof_probabilities"],
                }
            ).to_csv(folder / "oof_evaluation.csv", index=False)
            metadata = {
                "name": name,
                "feature_mode": mode,
                "strategy": strategy,
                "settings": settings,
                "device": str(device),
                "seed": seed,
                "fold_checksum": fold_digest,
                "development_checksum": checksum,
                "git_commit": commit,
                "git_dirty": dirty,
                "mean": result["mean"],
                "std": result["std"],
                "pooled": result["pooled"],
            }
            (folder / "configuration.json").write_text(json.dumps(metadata, indent=2))
            with mlflow.start_run(
                run_name=name,
                tags={"model": "ann", "feature_mode": mode, "strategy": strategy},
            ) as run:
                result["run_id"] = run.info.run_id
                mlflow.log_params(
                    {
                        "model": "ann",
                        "feature_mode": mode,
                        "loss_strategy": strategy,
                        "device": str(device),
                        "seed": seed,
                        "cv_folds": len(folds),
                        "fold_checksum": fold_digest,
                        "development_sha256": checksum,
                        "git_commit": commit,
                        "git_dirty": dirty,
                        "threshold": config["model_training"]["threshold"],
                        **{
                            f"ann__{key}": value
                            for key, value in settings.items()
                            if key not in {"feature_modes", "loss_strategies"}
                        },
                    }
                )
                for index, row in result["fold_metrics"].iterrows():
                    mlflow.log_metrics(
                        {
                            f"fold_{index + 1}_{key}": float(row[key])
                            for key in (*SCORES, "best_epoch", "training_seconds")
                        }
                    )
                    if strategy == "weighted":
                        mlflow.log_metrics(
                            {
                                f"fold_{index + 1}_inner_pos_weight": float(
                                    row.inner_pos_weight
                                ),
                                f"fold_{index + 1}_refit_pos_weight": float(
                                    row.refit_pos_weight
                                ),
                            }
                        )
                mlflow.log_metrics(
                    {f"cv_mean_{key}": value for key, value in result["mean"].items()}
                )
                mlflow.log_metrics(
                    {f"cv_std_{key}": value for key, value in result["std"].items()}
                )
                mlflow.log_metrics(
                    {f"oof_{key}": value for key, value in result["pooled"].items()}
                )
                mlflow.log_metrics(
                    {"best_epoch_mean": float(result["fold_metrics"].best_epoch.mean())}
                )
                mlflow.log_artifacts(str(folder))
            results.append(result)

    board = (
        pd.DataFrame(
            [
                {
                    "name": r["name"],
                    "model": "ann",
                    "feature_mode": r["feature_mode"],
                    "strategy": r["strategy"],
                    "ap_mean": r["mean"]["average_precision"],
                    "ap_std": r["std"]["average_precision"],
                    "roc_auc_mean": r["mean"]["roc_auc"],
                    "precision_mean": r["mean"]["precision"],
                    "recall_mean": r["mean"]["recall"],
                    "f1_mean": r["mean"]["f1"],
                    "balanced_accuracy_mean": r["mean"]["balanced_accuracy"],
                    "training_seconds_mean": r["mean"]["training_seconds"],
                    "best_epoch_mean": float(r["fold_metrics"].best_epoch.mean()),
                    "run_id": r["run_id"],
                }
                for r in results
            ]
        )
        .sort_values("ap_mean", ascending=False)
        .reset_index(drop=True)
    )
    board.to_csv(report / "leaderboard.csv", index=False)
    all_folds = []
    for result in results:
        frame = result["fold_metrics"].assign(name=result["name"])
        for column in ("inner_pos_weight", "refit_pos_weight"):
            frame[column] = frame[column].astype(float)
        all_folds.append(frame)
    pd.concat(all_folds).to_csv(report / "fold_metrics.csv", index=False)
    combined = pd.concat(
        [classical.assign(source="classical"), board.assign(source="ann")],
        ignore_index=True,
    )
    combined.sort_values("ap_mean", ascending=False).to_csv(
        report / "combined_leaderboard.csv", index=False
    )
    winner = max(results, key=lambda r: r["mean"]["average_precision"])
    plots = [
        _plot_history(winner, report),
        _plot_comparison(winner, y, classical, classical_root, report),
    ]
    with mlflow.start_run(run_id=winner["run_id"]):
        for path in plots:
            mlflow.log_artifact(str(path), artifact_path="summary")
        mlflow.log_artifact(
            str(report / "combined_leaderboard.csv"), artifact_path="summary"
        )
    summary = {
        "development_rows": len(y),
        "device": str(device),
        "fold_checksum": fold_digest,
        "development_checksum": checksum,
        "git_commit": commit,
        "git_dirty": dirty,
        "experiments": len(results),
        "ann_leader": winner["name"],
        "ann_leader_ap": winner["mean"]["average_precision"],
        "classical_leader": classical.iloc[0]["name"],
        "classical_leader_ap": float(classical.iloc[0].ap_mean),
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    print(run_ann_experiments(args.config).to_string(index=False))


if __name__ == "__main__":
    main()
