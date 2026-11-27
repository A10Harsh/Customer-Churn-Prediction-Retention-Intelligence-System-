"""Database contract checks that do not touch an application database."""

import pytest
from sqlalchemy.dialects.postgresql import JSONB

from retention.database.connection import database_url
from retention.database.models import Customer, Prediction
from retention.database.validation import validate_predictors


def test_database_url_requires_postgres_and_isolated_test_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    with pytest.raises(ValueError):
        database_url()
    monkeypatch.setenv("DATABASE_URL", "sqlite:///app.db")
    with pytest.raises(ValueError):
        database_url()
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost/app")
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+psycopg://u:p@localhost/app")
    with pytest.raises(ValueError):
        database_url(test=True)
    monkeypatch.setenv(
        "TEST_DATABASE_URL", "postgresql+psycopg://u:p@localhost/app_test"
    )
    assert database_url(test=True).endswith("app_test")


def test_postgresql_contract():
    assert isinstance(Prediction.__table__.c.predictor_snapshot.type, JSONB)
    assert (
        next(iter(Prediction.__table__.c.customer_id.foreign_keys)).ondelete
        == "RESTRICT"
    )
    assert "Churn" not in Customer.__table__.columns
    assert "customerID" not in Prediction.__table__.columns
    assert len(Prediction.__table__.indexes) == 2


def test_predictor_contract_rejects_target_and_invalid_values():
    from tests.integration.test_database import synthetic_customer

    values = synthetic_customer()
    assert validate_predictors(values)["tenure"] == 12
    with pytest.raises(ValueError):
        validate_predictors(dict(values, Churn="No"))
    with pytest.raises(ValueError):
        validate_predictors(dict(values, MonthlyCharges=float("nan")))
    with pytest.raises(ValueError):
        validate_predictors(dict(values, SeniorCitizen=2))


def test_session_scope_commits_rolls_back_and_closes():
    from retention.database.connection import session_scope

    class FakeSession:
        def __init__(self):
            self.events = []

        def commit(self):
            self.events.append("commit")

        def rollback(self):
            self.events.append("rollback")

        def close(self):
            self.events.append("close")

    successful = FakeSession()
    with session_scope(lambda: successful):
        pass
    assert successful.events == ["commit", "close"]

    failed = FakeSession()
    with pytest.raises(RuntimeError):
        with session_scope(lambda: failed):
            raise RuntimeError("synthetic failure")
    assert failed.events == ["rollback", "close"]


def test_input_guards_before_database_access():
    from retention.database.repository import _page, _unit_decimal, get_prediction

    with pytest.raises(ValueError):
        _page(101, 0)
    with pytest.raises(ValueError):
        _page(1, -1)
    with pytest.raises(ValueError):
        _unit_decimal(float("nan"), "probability")
    with pytest.raises(ValueError):
        _unit_decimal(1.01, "probability")
    with pytest.raises(ValueError):
        get_prediction(None, "not-a-uuid")


def test_connection_failure_disposes_engine_without_leaking_url(monkeypatch):
    from sqlalchemy.exc import OperationalError

    from retention.database import connection

    class FailedEngine:
        def __init__(self):
            self.disposed = False

        def connect(self):
            raise OperationalError("SELECT 1", None, Exception("unavailable"))

        def dispose(self):
            self.disposed = True

    engine = FailedEngine()
    monkeypatch.setattr(connection, "create_engine", lambda *args, **kwargs: engine)
    with pytest.raises(ConnectionError, match="Database connection failed"):
        connection.create_database_engine(
            "postgresql+psycopg://u:private@localhost/app"
        )
    assert engine.disposed


def test_service_consistency_and_numeric_precision():
    from tests.integration.test_database import synthetic_customer

    values = synthetic_customer()
    with pytest.raises(ValueError):
        validate_predictors(dict(values, PhoneService="No"))
    with pytest.raises(ValueError):
        validate_predictors(dict(values, InternetService="No"))
    with pytest.raises(ValueError):
        validate_predictors(dict(values, tenure=1.5))
    with pytest.raises(ValueError):
        validate_predictors(dict(values, MonthlyCharges="55.50001"))


def test_test_engine_rejects_application_database_even_with_other_host(monkeypatch):
    from retention.database.connection import create_database_engine

    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://app:private@localhost/retention_test"
    )
    with pytest.raises(ValueError, match="separate _test database"):
        create_database_engine(
            "postgresql+psycopg://test:private@otherhost/retention_test",
            test=True,
        )
    with pytest.raises(ValueError, match="postgresql\\+psycopg"):
        create_database_engine("sqlite:///local.db")


def test_rejects_invalid_categorical_and_negative_predictors():
    from tests.integration.test_database import synthetic_customer

    values = synthetic_customer()
    with pytest.raises(ValueError, match="Invalid category"):
        validate_predictors(dict(values, Contract="Unknown"))
    with pytest.raises(ValueError, match="Invalid numeric predictor"):
        validate_predictors(dict(values, TotalCharges=-1))
    with pytest.raises(ValueError, match="Invalid numeric predictor"):
        validate_predictors(dict(values, SeniorCitizen=True))
