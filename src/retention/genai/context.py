"""Minimal context from immutable prediction and explanation records."""

import math

from retention.genai.policies import eligible_actions

ALLOWED = {
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
}


def build_context(prediction, explanation, policies: dict) -> dict:
    if (
        explanation.prediction_id != prediction.id
        or explanation.model_fingerprint != prediction.model_fingerprint
        or explanation.explained_output_space != "raw_xgboost_margin"
        or abs(
            float(explanation.calibrated_probability)
            - float(prediction.churn_probability)
        )
        > 1e-12
    ):
        raise ValueError("Explanation provenance does not match prediction")
    probability = float(prediction.churn_probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid calibrated probability")
    attributes = {
        k: v for k, v in prediction.predictor_snapshot.items() if k in ALLOWED
    }
    if (
        not {"tenure", "Contract", "TechSupport", "InternetService", "OnlineSecurity"}
        <= attributes.keys()
    ):
        raise ValueError("Incomplete prediction snapshot")
    factors = []
    for direction, rows in (
        ("increasing", explanation.top_risk_increasing),
        ("decreasing", explanation.top_risk_decreasing),
    ):
        for row in rows:
            feature = row["feature"]
            value = float(row["raw_margin_contribution"])
            if not math.isfinite(value):
                raise ValueError("Nonfinite SHAP contribution")
            if feature in ALLOWED:
                factors.append(
                    {
                        "ref": f"shap:{feature}",
                        "feature": feature,
                        "direction": direction,
                        "raw_margin_contribution": value,
                    }
                )
    return {
        "probability": probability,
        "predicted_churn": prediction.predicted_churn,
        "attributes": attributes,
        "factors": factors,
        "eligible_actions": eligible_actions(
            attributes, prediction.predicted_churn, policies
        ),
        "semantics": (
            "SHAP values are raw-margin predictive attributions, "
            "not causal effects or probability changes."
        ),
        "restrictions": (
            "No fabricated history, complaints, discounts, prices, savings "
            "or outcomes. Human approval before intervention."
        ),
    }
