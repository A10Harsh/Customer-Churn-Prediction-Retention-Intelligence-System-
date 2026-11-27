"""Development-only sigmoid calibration with fold-isolated fitting."""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def validate_probabilities(values: object) -> np.ndarray:
    """Validate and return a one-dimensional probability array."""
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError("Probabilities must be a nonempty 1-D array.")
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return probabilities


def fit_sigmoid(probabilities: object, labels: object) -> LogisticRegression:
    """Fit Platt-style sigmoid to training-only OOF probabilities."""
    scores = validate_probabilities(probabilities)
    truth = np.asarray(labels, dtype=int)
    if truth.shape != scores.shape or set(np.unique(truth)) != {0, 1}:
        raise ValueError("Calibration labels must align and include both classes.")
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(scores.reshape(-1, 1), truth)
    return calibrator


def apply_sigmoid(calibrator: LogisticRegression, probabilities: object) -> np.ndarray:
    """Apply a fitted sigmoid calibrator to held-out probabilities."""
    scores = validate_probabilities(probabilities)
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]


def calibration_metrics(labels: object, probabilities: object) -> dict[str, float]:
    """Evaluate calibration and discrimination without a decision threshold."""
    scores = validate_probabilities(probabilities)
    truth = np.asarray(labels, dtype=int)
    if truth.shape != scores.shape or set(np.unique(truth)) != {0, 1}:
        raise ValueError("Evaluation labels must align and include both classes.")
    return {
        "brier": float(brier_score_loss(truth, scores)),
        "log_loss": float(log_loss(truth, scores, labels=[0, 1])),
        "average_precision": float(average_precision_score(truth, scores)),
        "roc_auc": float(roc_auc_score(truth, scores)),
    }


def reliability_points(
    labels: object, probabilities: object, bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Return observed and predicted bin rates for a reliability plot."""
    scores = validate_probabilities(probabilities)
    return calibration_curve(
        np.asarray(labels, dtype=int), scores, n_bins=bins, strategy="uniform"
    )
