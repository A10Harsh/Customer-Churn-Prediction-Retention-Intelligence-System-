"""Reusable path-dependent Tree SHAP explanations of the frozen XGBoost margin."""

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
import shap
from scipy.special import expit

from retention.database.inference import (
    ARTIFACT,
    FROZEN_THRESHOLD,
    MODEL_NAME,
    MODEL_VERSION,
    frozen_predictor,
)
from retention.database.validation import validate_predictors
from retention.evaluation.calibration import apply_sigmoid
from retention.explainability.aggregation import (
    aggregate_contributions,
    original_feature_map,
)
from retention.explainability.schemas import (
    CustomerExplanation,
    EncodedContribution,
    FeatureContribution,
)
from retention.models.predictor import ChurnPredictor

METHOD = "tree_shap"
VERSION = "1"
PERTURBATION = "tree_path_dependent"
OUTPUT_SPACE = "raw_xgboost_margin"
MAX_BATCH_SIZE = 100
LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior-citizen status",
    "Partner": "Partner status",
    "Dependents": "Dependent status",
    "tenure": "Tenure",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Technical support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
}
IMMUTABLE = {"gender", "SeniorCitizen", "Partner", "Dependents"}
HISTORICAL = {"tenure", "TotalCharges"}


def _json_value(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _factor(name: str, value, contribution: float) -> FeatureContribution:
    display = _json_value(value)
    if name == "Contract":
        subject = f"{display} contract"
    elif name == "tenure":
        subject = f"Tenure ({display} months)"
    else:
        subject = f"{LABELS[name]} ({display})"
    direction = (
        "increased"
        if contribution > 0
        else "reduced"
        if contribution < 0
        else "did not change"
    )
    description = f"{subject} {direction} the model's raw churn score."
    group = (
        "immutable_characteristic"
        if name in IMMUTABLE
        else "historical_context"
        if name in HISTORICAL
        else "service_or_billing"
    )
    return FeatureContribution(name, display, contribution, description, group)


def rank_factors(
    contributions: tuple[FeatureContribution, ...], top_k: int
) -> tuple[
    tuple[FeatureContribution, ...],
    tuple[FeatureContribution, ...],
    tuple[FeatureContribution, ...],
]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 19:
        raise ValueError("top_k must be between 1 and 19")
    positive = tuple(
        sorted(
            (item for item in contributions if item.raw_margin_contribution > 0),
            key=lambda item: (-item.raw_margin_contribution, item.feature),
        )[:top_k]
    )
    negative = tuple(
        sorted(
            (item for item in contributions if item.raw_margin_contribution < 0),
            key=lambda item: (item.raw_margin_contribution, item.feature),
        )[:top_k]
    )
    absolute = tuple(
        sorted(
            contributions,
            key=lambda item: (-abs(item.raw_margin_contribution), item.feature),
        )
    )
    return positive, negative, absolute


class ExplanationService:
    def __init__(self, predictor: ChurnPredictor, model_sha256: str):
        if predictor.threshold != FROZEN_THRESHOLD:
            raise ValueError("Frozen threshold mismatch")
        if len(model_sha256) != 64:
            raise ValueError("Model SHA-256 is required")
        self.predictor = predictor
        self.model_sha256 = model_sha256
        self.preprocessor = predictor.pipeline.named_steps["preprocess"]
        self.estimator = predictor.pipeline.named_steps["estimator"]
        self.encoded_names = tuple(self.preprocessor.get_feature_names_out())
        self.mapping = original_feature_map(self.preprocessor, predictor.columns)
        self.explainer = shap.TreeExplainer(
            self.estimator,
            model_output="raw",
            feature_perturbation=PERTURBATION,
        )
        fingerprint_input = {
            "method": METHOD,
            "version": VERSION,
            "perturbation": PERTURBATION,
            "output_space": OUTPUT_SPACE,
            "model_sha256": model_sha256,
            "configuration_sha256": predictor.metadata["configuration_sha256"],
            "shap_version": shap.__version__,
            "mapping": self.mapping,
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(fingerprint_input, sort_keys=True).encode()
        ).hexdigest()

    def explain(
        self, record: dict, *, prediction_id: str | None = None, top_k: int = 5
    ) -> CustomerExplanation:
        return self.explain_many([record], prediction_ids=[prediction_id], top_k=top_k)[
            0
        ]

    def explain_many(
        self,
        records: list[dict] | pd.DataFrame,
        *,
        prediction_ids: list[str | None] | None = None,
        top_k: int = 5,
    ) -> tuple[CustomerExplanation, ...]:
        if isinstance(records, pd.DataFrame):
            records = records.to_dict(orient="records")
        if not isinstance(records, list) or not 1 <= len(records) <= MAX_BATCH_SIZE:
            raise ValueError("Expected 1 to 100 customer records")
        if prediction_ids is None:
            prediction_ids = [None] * len(records)
        if len(prediction_ids) != len(records):
            raise ValueError("Prediction identifiers must align with records")
        for identifier in prediction_ids:
            if identifier is not None:
                UUID(str(identifier))
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 19
        ):
            raise ValueError("top_k must be between 1 and 19")
        normalized = [validate_predictors(item) for item in records]
        frame = self.predictor._validate(pd.DataFrame(normalized))
        transformed = np.asarray(self.preprocessor.transform(frame), dtype=float)
        if (
            transformed.shape != (len(records), len(self.encoded_names))
            or not np.isfinite(transformed).all()
        ):
            raise ValueError("Invalid transformed predictor matrix")
        try:
            shap_result = self.explainer(transformed, check_additivity=False)
        except Exception as exc:
            raise RuntimeError("Tree SHAP explanation failed") from exc
        contributions = np.asarray(shap_result.values, dtype=float)
        bases = np.asarray(shap_result.base_values, dtype=float).reshape(-1)
        margins = np.asarray(
            self.estimator.predict(transformed, output_margin=True), dtype=float
        ).reshape(-1)
        raw_probabilities = np.asarray(
            self.estimator.predict_proba(transformed)[:, 1], dtype=float
        )
        calibrated = apply_sigmoid(self.predictor.calibrator, raw_probabilities)
        if contributions.shape != transformed.shape or bases.shape != (len(records),):
            raise ValueError("Unexpected SHAP result dimensions")
        if not all(
            np.isfinite(array).all()
            for array in (contributions, bases, margins, raw_probabilities, calibrated)
        ):
            raise ValueError("Nonfinite explanation output")
        residuals = margins - bases - contributions.sum(axis=1)
        if not np.allclose(residuals, 0, atol=1e-4, rtol=0):
            raise ValueError("Tree SHAP local additivity failed")
        if not np.allclose(expit(margins), raw_probabilities, atol=1e-6, rtol=0):
            raise ValueError("Raw margin and model probability disagree")
        results = []
        for index, source in enumerate(normalized):
            original = aggregate_contributions(
                contributions[index], self.mapping, self.predictor.columns
            )
            factors = tuple(
                _factor(name, source[name], original[name])
                for name in self.predictor.columns
            )
            increasing, decreasing, absolute = rank_factors(factors, top_k)
            encoded = tuple(
                EncodedContribution(name, original_name, float(value))
                for name, original_name, value in zip(
                    self.encoded_names, self.mapping, contributions[index], strict=True
                )
            )
            result = CustomerExplanation(
                explanation_id=None,
                prediction_id=prediction_ids[index],
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                model_fingerprint=self.predictor.metadata["configuration_sha256"],
                explainer_fingerprint=self.fingerprint,
                explanation_method=METHOD,
                explanation_version=VERSION,
                feature_perturbation=PERTURBATION,
                explained_output_space=OUTPUT_SPACE,
                raw_prediction_margin=float(margins[index]),
                base_value=float(bases[index]),
                raw_churn_probability=float(raw_probabilities[index]),
                calibrated_churn_probability=float(calibrated[index]),
                predicted_churn=int(calibrated[index] >= FROZEN_THRESHOLD),
                threshold=FROZEN_THRESHOLD,
                original_feature_contributions=factors,
                transformed_feature_contributions=encoded,
                absolute_contribution_ranking=absolute,
                top_risk_increasing=increasing,
                top_risk_decreasing=decreasing,
                additivity_residual=float(residuals[index]),
                explanation_timestamp=datetime.now(UTC).isoformat(),
            )
            result.to_dict()
            results.append(result)
        return tuple(results)


@lru_cache(maxsize=1)
def frozen_explanation_service() -> ExplanationService:
    """Load and verify the trusted artifact once per process."""
    readiness = json.loads(Path("artifacts/final_model/readiness.json").read_text())
    actual_sha = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    if actual_sha != readiness["model_sha256"]:
        raise ValueError("Model artifact fingerprint mismatch")
    predictor = frozen_predictor()
    if predictor.metadata["configuration_sha256"] != readiness["configuration_sha256"]:
        raise ValueError("Frozen configuration fingerprint mismatch")
    return ExplanationService(predictor, actual_sha)
