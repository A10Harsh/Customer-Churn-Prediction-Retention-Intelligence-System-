"""Prediction-bound recommendation orchestration.

Phase 9B: LLM performs constrained action selection; deterministic renderer
generates all recommendation text from validated evidence and templates.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from retention.database.connection import session_scope
from retention.database.models import Explanation, RecommendationRecord
from retention.database.repository import get_prediction
from retention.genai.config import GenAIConfig, load_genai_config, load_policies
from retention.genai.context import build_context
from retention.genai.guardrails import validate_llm_selection, validate_recommendation
from retention.genai.llm import make_llm
from retention.genai.prompts import prompt_for
from retention.genai.renderer import RENDERING_VERSION, render_recommendation
from retention.genai.schemas import LLMSelection, ReasonCode, Recommendation


@dataclass(frozen=True)
class RecommendationResult:
    id: str
    prediction_id: str
    explanation_id: str
    recommendation: Recommendation
    generation_source: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None


def _result(row: RecommendationRecord) -> RecommendationResult:
    return RecommendationResult(
        str(row.id),
        str(row.prediction_id),
        str(row.explanation_id),
        Recommendation.model_validate(row.payload),
        row.generation_source,
        row.provider,
        row.model,
        row.latency_ms,
        row.input_tokens,
        row.output_tokens,
    )


def get_recommendation(
    session: Session, recommendation_id: UUID | str
) -> RecommendationResult | None:
    row = session.get(RecommendationRecord, UUID(str(recommendation_id)))
    return _result(row) if row else None


def recommendation_history(
    session: Session, prediction_id: UUID | str
) -> list[RecommendationResult]:
    rows = session.scalars(
        select(RecommendationRecord)
        .where(RecommendationRecord.prediction_id == UUID(str(prediction_id)))
        .order_by(
            RecommendationRecord.created_at.desc(), RecommendationRecord.id.desc()
        )
    )
    return [_result(row) for row in rows]


def _determine_reason_codes(action_id: str) -> list[str]:
    """Use only the two deterministic outcomes in the fallback path."""
    if action_id == "NO_ACTION":
        return [ReasonCode.RETAINED_CLASS]
    if action_id == "HUMAN_REVIEW":
        return [ReasonCode.INSUFFICIENT_EVIDENCE]
    raise ValueError("Only NO_ACTION and HUMAN_REVIEW are deterministic outcomes")


def _rule_recommendation(context: dict, *, fallback: bool) -> Recommendation:
    """Deterministic recommendation using the renderer."""
    if context["predicted_churn"] == 0:
        action_id = "NO_ACTION"
    else:
        action_id = "HUMAN_REVIEW"

    evidence_ids = ["prediction:risk"]
    reason_codes = _determine_reason_codes(action_id)
    policy_rule = context["eligible_actions"].get(action_id, {})

    rendered = render_recommendation(
        action_id=action_id,
        evidence_ids=evidence_ids,
        reason_codes=reason_codes,
        human_review=action_id == "HUMAN_REVIEW",
        context=context,
        policy_rule=policy_rule,
    )
    return validate_recommendation(rendered, context)


def _fingerprint(prediction, explanation, config: GenAIConfig, policies: dict) -> str:
    content = {
        "prediction_id": str(prediction.id),
        "explanation_id": str(explanation.id),
        "model_fingerprint": prediction.model_fingerprint,
        "explainer_fingerprint": explanation.explainer_fingerprint,
        "provider": config.provider,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "policy_version": config.policy_version,
        "rendering_version": RENDERING_VERSION,
        "policy_digest": hashlib.sha256(
            json.dumps(policies, sort_keys=True).encode()
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def recommend_prediction(
    factory: sessionmaker[Session],
    prediction_id: UUID | str,
    *,
    config: GenAIConfig | None = None,
    policies: dict | None = None,
    llm=None,
    regenerate_fallback: bool = False,
) -> RecommendationResult:
    config = config or load_genai_config()
    policies = policies or load_policies()
    if config.prompt_version != "3":
        raise ValueError("New recommendation generation requires prompt version 3")
    if config.policy_version != str(policies["version"]):
        raise ValueError("Policy version mismatch")
    with session_scope(factory) as session:
        prediction = get_prediction(session, prediction_id)
        if prediction is None:
            raise LookupError("Prediction not found")
        explanation = session.scalar(
            select(Explanation)
            .where(Explanation.prediction_id == prediction.id)
            .order_by(Explanation.created_at.desc(), Explanation.id.desc())
            .limit(1)
        )
        if explanation is None:
            raise LookupError("Prediction explanation not found")
        context = build_context(prediction, explanation, policies)
        fingerprint = _fingerprint(prediction, explanation, config, policies)
        existing = session.scalar(
            select(RecommendationRecord)
            .where(
                RecommendationRecord.prediction_id == prediction.id,
                RecommendationRecord.request_fingerprint == fingerprint,
            )
            .order_by(RecommendationRecord.attempt.desc())
            .limit(1)
        )
        if existing and (
            existing.generation_source != "fallback" or not regenerate_fallback
        ):
            return _result(existing)
        attempt = existing.attempt + 1 if existing else 1
        saved_prediction_id, saved_explanation_id = prediction.id, explanation.id
    start = time.perf_counter()
    source = "rules"
    usage = {}
    if context["predicted_churn"] == 0:
        recommendation = _rule_recommendation(context, fallback=False)
    else:
        try:
            client = llm if llm is not None else make_llm(config)
            response = client.with_structured_output(
                LLMSelection, include_raw=True
            ).invoke(prompt_for(context, config.prompt_version))
            selection = validate_llm_selection(response["parsed"], context)
            policy_rule = context["eligible_actions"][selection.action_id.value]
            rendered = render_recommendation(
                action_id=selection.action_id.value,
                evidence_ids=list(selection.evidence_ids),
                reason_codes=[rc.value for rc in selection.reason_codes],
                human_review=selection.human_review_required,
                context=context,
                policy_rule=policy_rule,
            )
            recommendation = validate_recommendation(rendered, context)
            source = "llm"
            usage = getattr(response["raw"], "usage_metadata", {}) or {}
        except Exception:
            if not config.fallback.enabled:
                raise
            recommendation = _rule_recommendation(context, fallback=True)
            source = "fallback"
    latency_ms = (time.perf_counter() - start) * 1000
    values = {
        "id": uuid4(),
        "prediction_id": saved_prediction_id,
        "explanation_id": saved_explanation_id,
        "action_id": recommendation.action_id.value,
        "payload": recommendation.model_dump(mode="json"),
        "provider": config.provider,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "policy_version": config.policy_version,
        "generation_source": source,
        "request_fingerprint": fingerprint,
        "attempt": attempt,
        "latency_ms": latency_ms,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }
    with session_scope(factory) as session:
        inserted = session.scalar(
            insert(RecommendationRecord)
            .values(**values)
            .on_conflict_do_nothing(constraint="uq_recommendations_request_attempt")
            .returning(RecommendationRecord.id)
        )
        row = (
            session.get(RecommendationRecord, inserted)
            if inserted
            else session.scalar(
                select(RecommendationRecord).where(
                    RecommendationRecord.prediction_id == saved_prediction_id,
                    RecommendationRecord.request_fingerprint == fingerprint,
                    RecommendationRecord.attempt == attempt,
                )
            )
        )
        if row is None:
            raise RuntimeError("Recommendation insert conflict could not be resolved")
        return _result(row)
