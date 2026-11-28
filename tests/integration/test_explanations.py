"""Live PostgreSQL explanation lifecycle using only synthetic test records."""

from sqlalchemy import func, select

from retention.database.connection import session_scope
from retention.database.explanations import (
    explain_prediction,
    get_explanation,
    get_prediction_explanation,
)
from retention.database.inference import predict_and_persist
from retention.database.models import Explanation, Prediction
from retention.database.repository import (
    customer_history,
    get_customer,
    insert_prediction,
    upsert_customer,
)
from retention.explainability.service import frozen_explanation_service
from tests.integration.test_database import synthetic_customer


def test_persistence_idempotency_and_snapshot_history(database):
    values = synthetic_customer()
    prediction_id, _ = predict_and_persist(database, "EXPLAIN-SYNTHETIC", values)
    service = frozen_explanation_service()
    first = explain_prediction(database, prediction_id, service=service)
    again = explain_prediction(database, prediction_id, service=service)
    assert first.to_dict() == again.to_dict()
    assert first.explanation_id is not None
    assert first.prediction_id == prediction_id
    assert first.model_fingerprint == service.predictor.metadata["configuration_sha256"]
    assert abs(first.additivity_residual) < 1e-4
    assert first.original_feature_contributions[0].feature == "gender"
    assert len(first.top_risk_increasing) <= 5
    other_top_k = explain_prediction(database, prediction_id, service=service, top_k=3)
    assert other_top_k.explanation_id != first.explanation_id
    assert len(other_top_k.top_risk_increasing) <= 3
    with session_scope(database) as session:
        stored = get_explanation(session, first.explanation_id)
        assert stored.prediction_id == session.get(Prediction, stored.prediction_id).id
        assert stored.model_fingerprint == first.model_fingerprint
        assert stored.explainer_fingerprint == first.explainer_fingerprint
        assert stored.full_payload == first.to_dict()
        assert (
            get_prediction_explanation(
                session, prediction_id, explainer_fingerprint=service.fingerprint
            ).id
            == stored.id
        )
        assert session.scalar(select(func.count()).select_from(Explanation)) == 2
        assert len(customer_history(session, "EXPLAIN-SYNTHETIC")) == 1
        assert (
            session.get(Prediction, stored.prediction_id).predictor_snapshot["tenure"]
            == 12
        )


def test_version_guard_and_explainer_failure_roll_back(database, monkeypatch):
    values = synthetic_customer()
    service = frozen_explanation_service()
    with session_scope(database) as session:
        customer = upsert_customer(session, "EXPLAIN-INCOMPATIBLE", values)
        incompatible = insert_prediction(
            session,
            customer=customer,
            probability=0.1,
            predicted_churn=0,
            threshold=service.predictor.threshold,
            model_name="customer-churn-predictor",
            model_version="2",
            model_fingerprint=service.predictor.metadata["configuration_sha256"],
            snapshot=values,
        )
        incompatible_id = incompatible.id
    import pytest

    with pytest.raises(ValueError, match="incompatible"):
        explain_prediction(database, incompatible_id, service=service)
    valid_id, _ = predict_and_persist(database, "EXPLAIN-ROLLBACK", values)
    before = None
    with session_scope(database) as session:
        before = session.scalar(select(func.count()).select_from(Explanation))

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic SHAP failure")

    monkeypatch.setattr(service, "explain", fail)
    with pytest.raises(RuntimeError, match="synthetic SHAP failure"):
        explain_prediction(database, valid_id, service=service)
    with session_scope(database) as session:
        assert session.scalar(select(func.count()).select_from(Explanation)) == before
        assert get_customer(session, "EXPLAIN-ROLLBACK") is not None
        assert len(customer_history(session, "EXPLAIN-ROLLBACK")) == 1
