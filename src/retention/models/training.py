"""Development-only classical-model CV and local MLflow experiment logging."""

import argparse
import hashlib
import json
import subprocess
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from retention.evaluation.metrics import binary_metrics, positive_probabilities
from retention.features.preprocessing import split_features_target
from retention.models.factory import make_estimator_pipeline
from retention.utils.config import load_config
from retention.utils.logger import get_logger

LOGGER = get_logger(__name__)
SCORES = (
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
    "accuracy",
    "specificity",
    "training_seconds",
    "inference_seconds",
)


def experiment_specs(config: dict) -> list[tuple[str, str, str]]:
    """Return one dummy and the twelve requested model comparisons."""
    modes = config["model_training"]["feature_modes"]
    strategies = config["model_training"]["imbalance_strategies"]
    if modes != ["baseline", "engineered"] or strategies != ["unweighted", "weighted"]:
        raise ValueError("Expected baseline/engineered and unweighted/weighted modes.")
    return [("dummy", "baseline", "unweighted")] + [
        (model, mode, strategy)
        for model in ("logistic_regression", "random_forest", "xgboost")
        for mode in modes
        for strategy in strategies
    ]


def make_folds(
    y: pd.Series, count: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate fixed stratified row-position folds once for all experiments."""
    return list(
        StratifiedKFold(n_splits=count, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        )
    )


def cross_validate_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    spec: tuple[str, str, str],
    config: dict,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config_path: str = "params.yaml",
) -> dict:
    """Fit a fresh full pipeline in each fold and reconcile complete OOF scores."""
    model, mode, strategy = spec
    seed = config["project"]["random_state"]
    threshold = config["model_training"]["threshold"]
    parameters = config["model_training"]["models"].get(model, {})
    probabilities = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_rows = []
    for number, (train_idx, valid_idx) in enumerate(folds, start=1):
        if (fold_ids[valid_idx] != -1).any():
            raise ValueError("Validation folds overlap.")
        train_y = y.iloc[train_idx]
        counts = train_y.value_counts()
        weight = (
            float(counts[0] / counts[1])
            if strategy == "weighted" and model == "xgboost"
            else None
        )
        pipeline = make_estimator_pipeline(
            model, mode, strategy, seed, parameters, weight, config_path
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            started = time.perf_counter()
            pipeline.fit(X.iloc[train_idx], train_y)
            train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        fold_probability = positive_probabilities(pipeline, X.iloc[valid_idx])
        inference_seconds = time.perf_counter() - started
        probabilities[valid_idx] = fold_probability
        fold_ids[valid_idx] = number
        metrics = binary_metrics(y.iloc[valid_idx], fold_probability, threshold)
        fold_rows.append(
            {
                "fold": number,
                "train_rows": len(train_idx),
                "validation_rows": len(valid_idx),
                "scale_pos_weight": weight,
                "training_seconds": train_seconds,
                "inference_seconds": inference_seconds,
                **metrics,
            }
        )
    if (fold_ids < 0).any() or not np.isfinite(probabilities).all():
        raise ValueError("Every observation must receive exactly one OOF probability.")
    folds_frame = pd.DataFrame(fold_rows)
    pooled = binary_metrics(y, probabilities, threshold)
    means = {metric: float(folds_frame[metric].mean()) for metric in SCORES}
    stds = {metric: float(folds_frame[metric].std(ddof=1)) for metric in SCORES}
    return {
        "fold_metrics": folds_frame,
        "oof_probabilities": probabilities,
        "fold_ids": fold_ids,
        "pooled": pooled,
        "mean": means,
        "std": stds,
    }


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    return commit, dirty


def _plot_summary(results: list[dict], y: pd.Series, output: Path) -> list[Path]:
    """Create only one comparison, one PR, and one confusion-matrix figure."""
    leaderboard = sorted(
        results, key=lambda r: r["mean"]["average_precision"], reverse=True
    )
    labels = [r["name"] for r in leaderboard]
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(labels))
    ax.barh(
        positions - 0.2,
        [r["mean"]["average_precision"] for r in leaderboard],
        0.4,
        label="AP",
    )
    ax.barh(
        positions + 0.2,
        [r["mean"]["roc_auc"] for r in leaderboard],
        0.4,
        label="ROC-AUC",
    )
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    comparison = output / "ap_roc_comparison.png"
    fig.savefig(comparison, dpi=130)
    plt.close(fig)

    leaders = [
        max(
            (r for r in results if r["model"] == model),
            key=lambda r: r["mean"]["average_precision"],
        )
        for model in ("dummy", "logistic_regression", "random_forest", "xgboost")
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    for result in leaders:
        precision, recall, _ = precision_recall_curve(y, result["oof_probabilities"])
        ax.plot(
            recall,
            precision,
            label=f"{result['name']} (AP={result['pooled']['average_precision']:.3f})",
        )
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7)
    fig.tight_layout()
    pr_path = output / "leading_precision_recall.png"
    fig.savefig(pr_path, dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, result in zip(axes, leaders[1:], strict=True):
        values = result["pooled"]
        matrix = np.array(
            [[values["tn"], values["fp"]], [values["fn"], values["tp"]]], dtype=int
        )
        ConfusionMatrixDisplay(matrix, display_labels=["No", "Yes"]).plot(
            ax=ax, colorbar=False, values_format="d"
        )
        ax.set_title(result["name"], fontsize=8)
    fig.tight_layout()
    confusion = output / "leading_confusion_matrices.png"
    fig.savefig(confusion, dpi=130)
    plt.close(fig)
    return [comparison, pr_path, confusion]


def run_training(config_path: str = "params.yaml") -> pd.DataFrame:
    """Execute and log all thirteen configured development-only experiments."""
    config = load_config(config_path)
    data_path = Path(config["data"]["development_path"])
    if data_path.resolve() == Path(config["data"]["test_path"]).resolve():
        raise ValueError("Development path must differ from held-out test path.")
    data = pd.read_parquet(data_path)
    X, y = split_features_target(data, config_path)
    seed = config["project"]["random_state"]
    folds = make_folds(y, config["validation"]["cv_folds"], seed)
    settings = config["model_training"]
    report = Path(settings["report_directory"])
    report.mkdir(parents=True, exist_ok=True)
    checksum = _checksum(data_path)
    commit, dirty = _git_state()
    mlflow.set_tracking_uri(settings["mlflow_tracking_uri"])
    mlflow.set_experiment(settings["mlflow_experiment"])
    results = []
    for spec in experiment_specs(config):
        model, mode, strategy = spec
        name = f"{model}__{mode}__{strategy}"
        LOGGER.info("Training %s", name)
        result = cross_validate_experiment(X, y, spec, config, folds, config_path)
        result.update(name=name, model=model, feature_mode=mode, strategy=strategy)
        experiment_dir = report / name
        experiment_dir.mkdir(exist_ok=True)
        result["fold_metrics"].to_csv(experiment_dir / "fold_metrics.csv", index=False)
        pd.DataFrame(
            {
                "fold": result["fold_ids"],
                "observed_churn": y.to_numpy(),
                "oof_probability": result["oof_probabilities"],
            }
        ).to_csv(experiment_dir / "oof_evaluation.csv", index=False)
        metadata = {
            "name": name,
            "model": model,
            "feature_mode": mode,
            "strategy": strategy,
            "hyperparameters": settings["models"].get(model, {}),
            "seed": seed,
            "cv_folds": len(folds),
            "threshold": settings["threshold"],
            "development_checksum": checksum,
            "git_commit": commit,
            "git_dirty": dirty,
            "mean": result["mean"],
            "std": result["std"],
            "pooled": result["pooled"],
        }
        (experiment_dir / "configuration.json").write_text(
            json.dumps(metadata, indent=2)
        )
        with mlflow.start_run(
            run_name=name,
            tags={"model": model, "feature_mode": mode, "strategy": strategy},
        ) as run:
            result["run_id"] = run.info.run_id
            mlflow.log_params(
                {
                    "model": model,
                    "feature_mode": mode,
                    "imbalance_strategy": strategy,
                    "preprocessing": "linear"
                    if model == "logistic_regression"
                    else "tree",
                    "random_seed": seed,
                    "cv_folds": len(folds),
                    "threshold": settings["threshold"],
                    "development_sha256": checksum,
                    "git_commit": commit,
                    "git_dirty": dirty,
                    **{
                        f"model__{key}": value
                        for key, value in settings["models"].get(model, {}).items()
                    },
                }
            )
            for index, row in result["fold_metrics"].iterrows():
                mlflow.log_metrics(
                    {f"fold_{index + 1}_{key}": float(row[key]) for key in SCORES}
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
            mlflow.log_artifacts(str(experiment_dir))
        results.append(result)

    rows = []
    for result in results:
        rows.append(
            {
                "name": result["name"],
                "model": result["model"],
                "feature_mode": result["feature_mode"],
                "strategy": result["strategy"],
                "ap_mean": result["mean"]["average_precision"],
                "ap_std": result["std"]["average_precision"],
                "roc_auc_mean": result["mean"]["roc_auc"],
                "precision_mean": result["mean"]["precision"],
                "recall_mean": result["mean"]["recall"],
                "f1_mean": result["mean"]["f1"],
                "balanced_accuracy_mean": result["mean"]["balanced_accuracy"],
                "training_seconds_mean": result["mean"]["training_seconds"],
                "run_id": result["run_id"],
            }
        )
    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("ap_mean", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard.to_csv(report / "leaderboard.csv", index=False)
    all_folds = []
    for result in results:
        frame = result["fold_metrics"].assign(name=result["name"])
        frame["scale_pos_weight"] = frame["scale_pos_weight"].astype(float)
        all_folds.append(frame)
    pd.concat(all_folds).to_csv(report / "fold_metrics.csv", index=False)
    plots = _plot_summary(results, y, report)
    with mlflow.start_run(run_id=leaderboard.iloc[0]["run_id"]):
        for path in plots:
            mlflow.log_artifact(str(path), artifact_path="summary")
        mlflow.log_artifact(str(report / "leaderboard.csv"), artifact_path="summary")
    summary = {
        "development_rows": len(y),
        "positive_rate": float(y.mean()),
        "development_checksum": checksum,
        "git_commit": commit,
        "git_dirty": dirty,
        "primary_metric": settings["primary_metric"],
        "threshold": settings["threshold"],
        "experiments": len(results),
        "leader": leaderboard.iloc[0]["name"],
        "dummy_ap_mean": next(
            r["mean"]["average_precision"] for r in results if r["model"] == "dummy"
        ),
    }
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    return leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    board = run_training(args.config)
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()
