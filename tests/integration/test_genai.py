"""Synthetic prediction through explanation and recommendation persistence."""

import pytest

from retention.database.connection import session_scope
from retention.database.explanations import explain_prediction
from retention.database.inference import predict_and_persist
from retention.database.models import RecommendationRecord
from retention.database.repository import get_prediction
from retention.genai.schemas import LLMSelection, Recommendation
from retention.genai.service import recommend_prediction, recommendation_history
from tests.integration.test_database import synthetic_customer


class FailingModel:
    def with_structured_output(self, schema, include_raw=False):
        return self

    def invoke(self, prompt):
        raise RuntimeError("Synthetic provider failure")


def test_synthetic_recommendation_idempotency_and_regeneration(database):
    values = synthetic_customer()
    values.update(
        tenure=1,
        Contract="Month-to-month",
        TechSupport="No",
        OnlineSecurity="No",
        MonthlyCharges="95.00",
        TotalCharges="95.00",
    )
    prediction_id, _ = predict_and_persist(database, "NBA-SYNTHETIC", values)
    explanation = explain_prediction(database, prediction_id)
    first = recommend_prediction(database, prediction_id, llm=FailingModel())
    second = recommend_prediction(database, prediction_id, llm=FailingModel())
    assert first.id == second.id
    assert first.explanation_id == explanation.explanation_id
    assert first.generation_source in {"rules", "fallback"}
    with session_scope(database) as session:
        prediction = get_prediction(session, prediction_id)
        assert prediction.model_fingerprint == explanation.model_fingerprint
        assert len(recommendation_history(session, prediction_id)) == 1
    if first.generation_source == "fallback":
        third = recommend_prediction(
            database, prediction_id, llm=FailingModel(), regenerate_fallback=True
        )
        assert third.id != first.id
        with session_scope(database) as session:
            assert len(recommendation_history(session, prediction_id)) == 2
            assert (
                session.query(RecommendationRecord)
                .filter_by(prediction_id=prediction.id)
                .count()
                == 2
            )


class ValidModelV3:
    """Mock that returns a constrained LLMSelection for prompt v3."""

    def with_structured_output(self, schema, include_raw=False):
        assert schema is LLMSelection
        assert include_raw
        return self

    def invoke(self, prompt):
        assert "NBA-SYNTHETIC" not in str(prompt)
        return {
            "parsed": LLMSelection.model_validate(
                {
                    "action_id": "CONTRACT_REVIEW",
                    "evidence_ids": ["attr:Contract", "prediction:risk"],
                    "reason_codes": ["MONTH_TO_MONTH", "PREDICTED_CHURN_RISK"],
                    "human_review_required": True,
                }
            ),
            "raw": type(
                "Raw",
                (),
                {"usage_metadata": {"input_tokens": 80, "output_tokens": 30}},
            )(),
        }


class InvalidEvidenceModelV3(ValidModelV3):
    def invoke(self, prompt):
        result = super().invoke(prompt)
        result["parsed"] = result["parsed"].model_copy(
            update={"evidence_ids": ["prediction:risk"]}
        )
        return result


def test_synthetic_v3_structured_generation(database):
    """Test prompt v3 with constrained LLMSelection -> rendered recommendation."""
    values = synthetic_customer()
    values.update(
        tenure=1,
        Contract="Month-to-month",
        TechSupport="No",
        OnlineSecurity="No",
        MonthlyCharges="95.00",
        TotalCharges="95.00",
    )
    prediction_id, _ = predict_and_persist(database, "NBA-V3-SYNTHETIC", values)
    explain_prediction(database, prediction_id)
    from retention.genai.config import load_genai_config

    config = load_genai_config()
    assert config.prompt_version == "3"
    result = recommend_prediction(database, prediction_id, llm=ValidModelV3())
    assert result.generation_source == "llm"
    assert result.recommendation.action_id.value == "CONTRACT_REVIEW"
    assert result.input_tokens == 80
    assert result.output_tokens == 30
    # Verify rendered text is grounded
    assert "discount" not in result.recommendation.summary.lower()
    assert result.recommendation.rendering_version == "1"


def test_v3_invalid_evidence_fallback_and_retained_rule(database):
    risky = synthetic_customer()
    risky.update(
        tenure=1,
        Contract="Month-to-month",
        TechSupport="No",
        OnlineSecurity="No",
        MonthlyCharges="95.00",
        TotalCharges="95.00",
    )
    risk_id, _ = predict_and_persist(database, "NBA-V3-BAD-EVIDENCE", risky)
    explain_prediction(database, risk_id)
    fallback = recommend_prediction(database, risk_id, llm=InvalidEvidenceModelV3())
    assert fallback.generation_source == "fallback"
    assert fallback.recommendation.action_id.value == "HUMAN_REVIEW"
    assert fallback.recommendation.reason_codes[0].value == "INSUFFICIENT_EVIDENCE"

    retained = synthetic_customer()
    retained.update(
        tenure=60,
        Contract="Two year",
        TechSupport="Yes",
        OnlineSecurity="Yes",
        MonthlyCharges="45.00",
        TotalCharges="2700.00",
    )
    retained_id, prediction = predict_and_persist(database, "NBA-V3-RETAINED", retained)
    assert prediction["predicted_churn"] == 0
    explain_prediction(database, retained_id)
    result = recommend_prediction(database, retained_id, llm=FailingModel())
    assert result.generation_source == "rules"
    assert result.recommendation.action_id.value == "NO_ACTION"
    assert result.recommendation.reason_codes[0].value == "RETAINED_CLASS"


def test_legacy_payload_readable_but_new_generation_rejected():
    """Preserve historical rows without allowing new unrestricted prose."""
    from retention.genai.config import load_genai_config

    historical = Recommendation.model_validate(
        {
            "action_id": "CONTRACT_REVIEW",
            "priority": "medium",
            "channel": "phone",
            "summary": "Discuss contract options.",
            "rationale": "Observed month-to-month contract.",
            "evidence_refs": ["attr:Contract"],
            "talking_points": [],
            "human_approval_required": True,
            "limitations": "Predictive evidence is not causal.",
        }
    )
    assert historical.reason_codes == []
    assert historical.rendering_version is None
    config = load_genai_config()
    legacy_config = config.model_copy(update={"prompt_version": "2"})
    with pytest.raises(ValueError, match="prompt version 3"):
        recommend_prediction(
            None, "00000000-0000-0000-0000-000000000000", config=legacy_config
        )
