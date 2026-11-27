"""Trusted, self-contained inference artifact for the frozen churn model."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from retention.evaluation.calibration import apply_sigmoid
from retention.evaluation.metrics import positive_probabilities


@dataclass
class ChurnPredictor:
    pipeline: object
    calibrator: object
    threshold: float
    columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categories: dict[str, tuple[object, ...]]
    metadata: dict

    def _validate(self, records: pd.DataFrame | dict) -> pd.DataFrame:
        if isinstance(records, dict):
            records = pd.DataFrame([records])
        if not isinstance(records, pd.DataFrame) or records.empty:
            raise ValueError("Expected a nonempty DataFrame or one customer dictionary")
        missing = set(self.columns) - set(records)
        extra = set(records) - set(self.columns)
        if missing or extra:
            raise ValueError(
                f"Input schema mismatch: missing={sorted(missing)}; "
                f"extra={sorted(extra)}"
            )
        frame = records.loc[:, self.columns].copy()
        if frame.isna().any().any():
            raise ValueError("Missing predictor values are not supported")
        for name in self.numeric_columns:
            try:
                frame[name] = pd.to_numeric(frame[name], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric predictor: {name}") from exc
            if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite numeric predictor: {name}")
        for name, allowed in self.categories.items():
            if not frame[name].isin(allowed).all():
                raise ValueError(f"Unsupported category in {name}")
        return frame

    def raw_scores(self, records: pd.DataFrame | dict) -> np.ndarray:
        return positive_probabilities(self.pipeline, self._validate(records))

    def predict(self, records: pd.DataFrame | dict) -> pd.DataFrame:
        raw = self.raw_scores(records)
        probability = apply_sigmoid(self.calibrator, raw)
        if (
            not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()
        ):
            raise ValueError("Calibrated probabilities must be finite and in [0, 1]")
        return pd.DataFrame(
            {
                "churn_probability": probability,
                "predicted_churn": (probability >= self.threshold).astype(int),
                "threshold": np.full(len(probability), self.threshold),
            }
        )


def load_predictor(path: str | Path) -> ChurnPredictor:
    """Load only a locally trusted joblib artifact; joblib can execute code."""
    value = joblib.load(path)
    if not isinstance(value, ChurnPredictor):
        raise ValueError("Serialized object is not a ChurnPredictor")
    return value
