"""Deterministic development-only F1 threshold selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from retention.evaluation.calibration import validate_probabilities
from retention.evaluation.metrics import binary_metrics


def threshold_table(labels: object, probabilities: object) -> pd.DataFrame:
    """Evaluate every distinct decision boundary and the 0.5 reference."""
    scores = validate_probabilities(probabilities)
    truth = np.asarray(labels, dtype=int)
    if truth.shape != scores.shape or set(np.unique(truth)) != {0, 1}:
        raise ValueError("Threshold labels must align and include both classes.")
    candidates = np.unique(np.r_[scores[(scores > 0) & (scores < 1)], 0.5])
    rows = [
        {"threshold": float(value), **binary_metrics(truth, scores, float(value))}
        for value in candidates
    ]
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> dict:
    """Break F1 ties by recall, then the higher threshold."""
    if table.empty:
        raise ValueError("Threshold table is empty.")
    return (
        table.sort_values(
            ["f1", "recall", "threshold"], ascending=[False, False, False]
        )
        .iloc[0]
        .to_dict()
    )


def operating_points(labels: object, probabilities: object) -> pd.DataFrame:
    """Evaluate every attainable positive-prediction boundary efficiently."""
    scores = validate_probabilities(probabilities)
    truth = np.asarray(labels, dtype=int)
    if truth.shape != scores.shape or set(np.unique(truth)) != {0, 1}:
        raise ValueError("Threshold labels must align and include both classes.")
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    positives = int(truth.sum())
    negatives = len(truth) - positives
    tp = recall[:-1] * positives
    tp = np.rint(tp).astype(int)
    fp = np.rint(tp / np.maximum(precision[:-1], 1e-15) - tp).astype(int)
    predicted = tp + fp
    frame = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
            "tp": tp,
            "fp": fp,
            "tn": negatives - fp,
            "fn": positives - tp,
            "predicted_positive_proportion": predicted / len(truth),
        }
    )
    frame["f1"] = np.divide(
        2 * frame.precision * frame.recall,
        frame.precision + frame.recall,
    ).fillna(0)
    frame["specificity"] = frame.tn / negatives
    frame["balanced_accuracy"] = (frame.recall + frame.specificity) / 2
    frame["accuracy"] = (frame.tp + frame.tn) / len(truth)
    reference = binary_metrics(truth, scores, 0.5)
    reference["threshold"] = 0.5
    reference["predicted_positive_proportion"] = (
        reference["tp"] + reference["fp"]
    ) / len(truth)
    return pd.concat([frame, pd.DataFrame([reference])], ignore_index=True)


def select_constrained_recall(
    table: pd.DataFrame, minimum_precision: float = 0.55
) -> dict | None:
    """Maximize recall, then precision, F1, and threshold under a precision floor."""
    if not 0 < minimum_precision <= 1:
        raise ValueError("minimum_precision must be in (0, 1].")
    valid = table[(table.precision >= minimum_precision) & ((table.tp + table.fp) > 0)]
    if valid.empty:
        return None
    return (
        valid.sort_values(
            ["recall", "precision", "f1", "threshold"],
            ascending=[False, False, False, False],
        )
        .iloc[0]
        .to_dict()
    )
