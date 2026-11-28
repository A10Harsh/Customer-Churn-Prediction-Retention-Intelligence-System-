"""Prediction-bound explanation persistence with provenance and idempotency."""

from dataclasses import replace
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from retention.database.connection import session_scope
from retention.database.inference import FROZEN_THRESHOLD, MODEL_NAME, MODEL_VERSION
from retention.database.models import Explanation
from retention.database.repository import get_prediction
from retention.explainability.schemas import CustomerExplanation
from retention.explainability.service import (
    METHOD,
    OUTPUT_SPACE,
    VERSION,
    ExplanationService,
    frozen_explanation_service,
)


def get_explanation(session: Session, explanation_id: UUID | str) -> Explanation | None:
    try:
        parsed = UUID(str(explanation_id))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid explanation identifier") from exc
    return session.get(Explanation, parsed)


def get_prediction_explanation(
    session: Session,
    prediction_id: UUID | str,
    *,
    explainer_fingerprint: str,
    top_k: int = 5,
) -> Explanation | None:
    try:
        parsed = UUID(str(prediction_id))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid prediction identifier") from exc
    return session.scalar(
        select(Explanation).where(
            Explanation.prediction_id == parsed,
            Explanation.explanation_method == METHOD,
            Explanation.explanation_version == VERSION,
            Explanation.explainer_fingerprint == explainer_fingerprint,
            Explanation.top_k == top_k,
        )
    )


def _check_provenance(prediction, service: ExplanationService) -> None:
    if (
        prediction.model_name != MODEL_NAME
        or prediction.model_version != MODEL_VERSION
        or prediction.model_fingerprint
        != service.predictor.metadata["configuration_sha256"]
        or float(prediction.threshold) != FROZEN_THRESHOLD
    ):
        raise ValueError("Prediction and frozen explanation model are incompatible")


def explain_prediction(
    factory: sessionmaker[Session],
    prediction_id: UUID | str,
    *,
    service: ExplanationService | None = None,
    top_k: int = 5,
) -> CustomerExplanation:
    """Use the saved snapshot and return only after the database commit succeeds."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 19:
        raise ValueError("top_k must be between 1 and 19")
    explainer = service if service is not None else frozen_explanation_service()
    with session_scope(factory) as session:
        prediction = get_prediction(session, prediction_id)
        if prediction is None:
            raise LookupError("Prediction not found")
        _check_provenance(prediction, explainer)
        existing = get_prediction_explanation(
            session,
            prediction.id,
            explainer_fingerprint=explainer.fingerprint,
            top_k=top_k,
        )
        if existing is not None:
            return CustomerExplanation.from_dict(existing.full_payload)
        result = explainer.explain(
            prediction.predictor_snapshot,
            prediction_id=str(prediction.id),
            top_k=top_k,
        )
        if (
            result.model_fingerprint != prediction.model_fingerprint
            or result.model_version != prediction.model_version
            or result.explained_output_space != OUTPUT_SPACE
            or result.explanation_method != METHOD
            or result.explanation_version != VERSION
            or result.predicted_churn != prediction.predicted_churn
            or abs(
                result.calibrated_churn_probability
                - float(prediction.churn_probability)
            )
            > 1e-12
        ):
            raise ValueError("Explanation does not match saved prediction")
        explanation_id = uuid4()
        result = replace(result, explanation_id=str(explanation_id))
        payload = result.to_dict()
        statement = (
            insert(Explanation)
            .values(
                id=explanation_id,
                prediction_id=prediction.id,
                explanation_method=METHOD,
                explanation_version=VERSION,
                top_k=top_k,
                explained_output_space=OUTPUT_SPACE,
                base_value=result.base_value,
                raw_margin=result.raw_prediction_margin,
                raw_probability=Decimal(str(result.raw_churn_probability)),
                calibrated_probability=Decimal(
                    str(result.calibrated_churn_probability)
                ),
                additivity_residual=result.additivity_residual,
                full_payload=payload,
                original_feature_contributions=payload[
                    "original_feature_contributions"
                ],
                top_risk_increasing=payload["top_risk_increasing"],
                top_risk_decreasing=payload["top_risk_decreasing"],
                model_fingerprint=result.model_fingerprint,
                explainer_fingerprint=result.explainer_fingerprint,
            )
            .on_conflict_do_nothing(constraint="uq_explanations_request")
            .returning(Explanation.id)
        )
        inserted = session.scalar(statement)
        if inserted is None:
            stored = get_prediction_explanation(
                session,
                prediction.id,
                explainer_fingerprint=explainer.fingerprint,
                top_k=top_k,
            )
            if stored is None:
                raise RuntimeError("Explanation insert conflict could not be resolved")
            return CustomerExplanation.from_dict(stored.full_payload)
    return result
