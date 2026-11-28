"""Stable, strictly JSON-serializable explanation contracts."""

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: str | int | float
    raw_margin_contribution: float
    description: str
    attribute_group: str


@dataclass(frozen=True)
class EncodedContribution:
    encoded_feature: str
    original_feature: str
    raw_margin_contribution: float


@dataclass(frozen=True)
class CustomerExplanation:
    explanation_id: str | None
    prediction_id: str | None
    model_name: str
    model_version: str
    model_fingerprint: str
    explainer_fingerprint: str
    explanation_method: str
    explanation_version: str
    feature_perturbation: str
    explained_output_space: str
    raw_prediction_margin: float
    base_value: float
    raw_churn_probability: float
    calibrated_churn_probability: float
    predicted_churn: int
    threshold: float
    original_feature_contributions: tuple[FeatureContribution, ...]
    transformed_feature_contributions: tuple[EncodedContribution, ...]
    absolute_contribution_ranking: tuple[FeatureContribution, ...]
    top_risk_increasing: tuple[FeatureContribution, ...]
    top_risk_decreasing: tuple[FeatureContribution, ...]
    additivity_residual: float
    explanation_timestamp: str

    def to_dict(self) -> dict:
        """Reject non-JSON and nonfinite values."""
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value: dict) -> "CustomerExplanation":
        nested = dict(value)
        for key in (
            "original_feature_contributions",
            "absolute_contribution_ranking",
            "top_risk_increasing",
            "top_risk_decreasing",
        ):
            nested[key] = tuple(FeatureContribution(**item) for item in nested[key])
        nested["transformed_feature_contributions"] = tuple(
            EncodedContribution(**item)
            for item in nested["transformed_feature_contributions"]
        )
        result = cls(**nested)
        result.to_dict()
        return result
