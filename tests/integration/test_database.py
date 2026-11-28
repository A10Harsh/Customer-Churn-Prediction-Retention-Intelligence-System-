"""PostgreSQL-only tests against a separate, initially empty test database."""

from uuid import UUID

import pytest
from sqlalchemy.exc import IntegrityError

from retention.database.connection import (
    session_scope,
)
from retention.database.inference import predict_and_persist
from retention.database.repository import (
    customer_history,
    get_customer,
    get_prediction,
    insert_prediction,
    latest_prediction,
    list_customers,
    list_predictions,
    update_customer,
    upsert_customer,
)
from retention.models.predictor import load_predictor


def synthetic_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": "55.50",
        "TotalCharges": "666.00",
    }


def test_customer_upsert_history_and_provenance(database):
    values = synthetic_customer()
    predictor = load_predictor("artifacts/final_model/predictor.joblib")
    first_id, first_result = predict_and_persist(
        database, "SYNTHETIC-001", values, predictor=predictor
    )
    second_id, second_result = predict_and_persist(
        database, "SYNTHETIC-001", values, predictor=predictor
    )
    assert first_id != second_id
    assert first_result == second_result
    with session_scope(database) as session:
        customer = get_customer(session, "SYNTHETIC-001")
        assert customer.id == upsert_customer(session, "SYNTHETIC-001", values).id
        assert len(list_customers(session)) == 1
        assert len(list_predictions(session)) == 2
        history = customer_history(session, "SYNTHETIC-001")
        assert len(history) == 2
        assert latest_prediction(session, "SYNTHETIC-001").id == history[0].id
        prediction = get_prediction(session, UUID(first_id))
        assert prediction.customer_id == customer.id
        assert prediction.model_name == "customer-churn-predictor"
        assert prediction.model_version == "1"
        assert (
            prediction.model_fingerprint == predictor.metadata["configuration_sha256"]
        )
        assert float(prediction.threshold) == predictor.threshold
        assert "customerID" not in prediction.predictor_snapshot
        revised = dict(values, tenure=13)
        update_customer(session, "SYNTHETIC-001", revised)
        assert prediction.predictor_snapshot["tenure"] == 12


def test_validation_rollback_and_foreign_key(database):
    values = synthetic_customer()
    with pytest.raises(ValueError):
        with session_scope(database) as session:
            customer = upsert_customer(session, "ROLLBACK-001", values)
            insert_prediction(
                session,
                customer=customer,
                probability=float("nan"),
                predicted_churn=0,
                threshold=0.33,
                model_name="customer-churn-predictor",
                model_version="1",
                model_fingerprint="a" * 64,
                snapshot=values,
            )
    with session_scope(database) as session:
        assert get_customer(session, "ROLLBACK-001") is None
        customer = get_customer(session, "SYNTHETIC-001")
        with pytest.raises(ValueError):
            insert_prediction(
                session,
                customer=customer,
                probability=0.8,
                predicted_churn=0,
                threshold=0.33,
                model_name="customer-churn-predictor",
                model_version="1",
                model_fingerprint="a" * 64,
                snapshot=values,
            )
    with pytest.raises(IntegrityError):
        with session_scope(database) as session:
            customer = get_customer(session, "SYNTHETIC-001")
            session.delete(customer)
            session.flush()


def test_inference_from_stored_customer_and_missing_record(database):
    values = synthetic_customer()
    with session_scope(database) as session:
        upsert_customer(session, "SYNTHETIC-LOOKUP", values)
    prediction_id, result = predict_and_persist(database, "SYNTHETIC-LOOKUP")
    assert result["threshold"] == 0.3304499866713542
    with session_scope(database) as session:
        assert get_prediction(session, prediction_id) is not None
        assert len(customer_history(session, "SYNTHETIC-LOOKUP")) == 1
        assert update_customer(session, "MISSING-CUSTOMER", values) is None
        assert latest_prediction(session, "MISSING-CUSTOMER") is None
        assert customer_history(session, "MISSING-CUSTOMER") == []
        assert list_customers(session, limit=1, offset=100) == []
    with pytest.raises(LookupError):
        predict_and_persist(database, "MISSING-CUSTOMER")


def test_database_probability_constraint_rejects_direct_invalid_row(database):
    from decimal import Decimal

    from retention.database.models import Prediction

    with session_scope(database) as session:
        customer = get_customer(session, "SYNTHETIC-LOOKUP")
        before = len(customer_history(session, "SYNTHETIC-LOOKUP"))
    with pytest.raises(IntegrityError):
        with session_scope(database) as session:
            session.add(
                Prediction(
                    customer_id=customer.id,
                    churn_probability=Decimal("-0.1"),
                    predicted_churn=0,
                    threshold=Decimal("0.33"),
                    model_name="customer-churn-predictor",
                    model_version="1",
                    model_fingerprint="a" * 64,
                    predictor_snapshot=synthetic_customer(),
                )
            )
            session.flush()
    with session_scope(database) as session:
        assert len(customer_history(session, "SYNTHETIC-LOOKUP")) == before
