"""Shared binary classification metrics for fold and out-of-fold evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_probabilities(estimator: object, X: object) -> np.ndarray:
    """Return probabilities for class 1, regardless of class-column order."""
    classes = np.asarray(estimator.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError("Estimator must expose positive class 1.")
    values = np.asarray(estimator.predict_proba(X), dtype=float)[:, matches[0]]
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Predicted probabilities must be finite and in [0, 1].")
    return values


def binary_metrics(
    y_true: object, probability: object, threshold: float = 0.5
) -> dict[str, float]:
    """Compute AP, discrimination, and fixed-threshold classification scores."""
    truth = np.asarray(y_true, dtype=int)
    proba = np.asarray(probability, dtype=float)
    if truth.shape != proba.shape or truth.ndim != 1:
        raise ValueError("Labels and probabilities must be aligned 1-D arrays.")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("Both binary classes are required for evaluation.")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one.")
    if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    predicted = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(truth, proba)),
        "roc_auc": float(roc_auc_score(truth, proba)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "specificity": float(tn / (tn + fp)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
