"""Independent, immutable held-out evaluation of the frozen inference artifact."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_curve,
)

from retention.evaluation.diagnostics import json_safe, safe_metrics
from retention.evaluation.metrics import binary_metrics
from retention.models.predictor import load_predictor
from retention.models.training import _checksum
from retention.utils.config import load_config

ARTIFACT = Path("artifacts/final_model/predictor.joblib")
MANIFEST = Path("artifacts/final_model/readiness.json")
OUTPUT = Path("reports/generated/final_evaluation")
GOVERNANCE = Path("artifacts/final_model/heldout_evaluation.json")
METRICS = ("average_precision", "roc_auc", "precision", "recall", "f1")


def bootstrap_intervals(y, p, threshold, seed=42, resamples=2000):
    """Paired row percentile intervals conditional on the fitted model."""
    rng = np.random.default_rng(seed)
    values = {key: [] for key in METRICS}
    skipped = 0
    for _ in range(resamples):
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            skipped += 1
            continue
        row = binary_metrics(y[index], p[index], threshold)
        for key in METRICS:
            if np.isfinite(row[key]):
                values[key].append(row[key])
    return {
        key: {
            "lower": float(np.percentile(items, 2.5)) if items else None,
            "upper": float(np.percentile(items, 97.5)) if items else None,
            "valid_resamples": len(items),
        }
        for key, items in values.items()
    } | {
        "method": "paired-row percentile bootstrap",
        "resamples": resamples,
        "single_class_skipped": skipped,
        "seed": seed,
    }


def segment_metrics(X, y, p, threshold):
    frame = X.copy()
    frame["tenure_group"] = pd.cut(
        frame.tenure,
        [-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    )
    result = []
    for column in (
        "Contract",
        "tenure_group",
        "InternetService",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
    ):
        for value in frame[column].dropna().unique():
            mask = (frame[column] == value).to_numpy()
            row = safe_metrics(y[mask], p[mask], threshold)
            result.append({"segment": column, "value": str(value), **row})
    return result


def _figures(y, p, threshold, output):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    fig.savefig(output / "roc.png", bbox_inches="tight")
    plt.close(fig)
    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    fig.savefig(output / "precision_recall.png", bbox_inches="tight")
    plt.close(fig)
    predicted = (p >= threshold).astype(int)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y, predicted, labels=[0, 1], ax=ax)
    fig.savefig(output / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    observed, predicted_mean = calibration_curve(y, p, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.plot(predicted_mean, observed, marker="o")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed churn rate",
        title="Calibration",
    )
    fig.savefig(output / "calibration.png", bbox_inches="tight")
    plt.close(fig)


def evaluate(config_path="params.yaml", output=OUTPUT):
    config = load_config(config_path)
    manifest = json.loads(MANIFEST.read_text())
    predictor = load_predictor(ARTIFACT)
    if (
        manifest["readiness"] != "passed"
        or _checksum(ARTIFACT) != manifest["model_sha256"]
    ):
        raise ValueError("Training readiness or model fingerprint failed")
    if (
        predictor.metadata["configuration_sha256"] != manifest["configuration_sha256"]
        or _checksum(Path("configs/selected_model.yaml"))
        != manifest["configuration_sha256"]
    ):
        raise ValueError("Frozen configuration changed after training")
    if (
        predictor.metadata["development_sha256"] != manifest["development_sha256"]
        or _checksum(Path(config["data"]["development_path"]))
        != manifest["development_sha256"]
    ):
        raise ValueError("Development data changed after training")
    test_path = Path(config["data"]["test_path"])
    if test_path.name != "test.parquet":
        raise ValueError("Expected independent test.parquet")
    test_sha = _checksum(test_path)
    output = Path(output)
    record_path = output / "evaluation.json"
    if GOVERNANCE.exists():
        prior = json.loads(GOVERNANCE.read_text())
        if prior["fingerprints"] != {
            "model_sha256": manifest["model_sha256"],
            "test_sha256": test_sha,
            "configuration_sha256": manifest["configuration_sha256"],
        }:
            raise ValueError(
                "First held-out evaluation is immutable; fingerprints differ"
            )
        if record_path.exists():
            return prior
    data = pd.read_parquet(test_path)
    if len(data) != 1409:
        raise ValueError("Held-out row count mismatch")
    X = data.drop(columns=["Churn", "customerID"])
    y = data.Churn.map({"No": 0, "Yes": 1}).to_numpy()
    started = time.perf_counter()
    predictions = predictor.predict(X)
    elapsed = time.perf_counter() - started
    p = predictions.churn_probability.to_numpy()
    primary = binary_metrics(y, p, predictor.threshold)
    primary.update(
        {
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p)),
            "predicted_positive_proportion": float(predictions.predicted_churn.mean()),
            "batch_inference_seconds": elapsed,
            "precision_floor_met": bool(primary["precision"] >= 0.55),
        }
    )
    secondary = binary_metrics(y, p, 0.5)
    intervals = bootstrap_intervals(
        y, p, predictor.threshold, config["project"]["random_state"]
    )
    segments = segment_metrics(X, y, p, predictor.threshold)
    record = {
        "fingerprints": {
            "model_sha256": manifest["model_sha256"],
            "test_sha256": test_sha,
            "configuration_sha256": manifest["configuration_sha256"],
        },
        "sample_size": len(y),
        "threshold": predictor.threshold,
        "primary": primary,
        "reference_0_5": secondary,
        "bootstrap_95": intervals,
        "segments": segments,
        "development_comparison": {
            "pooled_calibrated_ap": 0.6630123846303676,
            "pooled_calibrated_roc_auc": 0.84927716694369,
            "precision": 0.5506112469437653,
            "recall": 0.7531772575250836,
            "f1": 0.6361581920903955,
        },
        "uncertainty_scope": (
            "Test-sample uncertainty conditional on the fitted model; "
            "not future temporal generalization"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    record = json_safe(record)
    record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if not GOVERNANCE.exists():
        GOVERNANCE.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    _figures(y, p, predictor.threshold, output)
    (output / "report.html").write_text(
        "<html><body><h1>Final held-out evaluation</h1><pre>"
        + json.dumps(
            {"primary": primary, "reference_0_5": secondary, "bootstrap_95": intervals},
            indent=2,
        )
        + "</pre><p>Intervals measure test-sample uncertainty conditional on "
        "the fitted model.</p></body></html>"
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    result = evaluate(args.config)
    print(json.dumps(result["primary"], indent=2))


if __name__ == "__main__":
    main()
