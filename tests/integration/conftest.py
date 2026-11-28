"""Isolated PostgreSQL migration fixture shared by integration tests."""

import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import MetaData, Table, inspect, text

from retention.database.connection import (
    create_database_engine,
    database_url,
    make_session_factory,
)


@pytest.fixture(scope="module")
def database():
    if not os.getenv("TEST_DATABASE_URL"):
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    database_url(test=True)
    engine = create_database_engine(test=True)
    tables = set(inspect(engine).get_table_names())
    if tables == {"alembic_version"}:
        with engine.connect() as connection:
            empty_version = (
                connection.scalar(text("SELECT count(*) FROM alembic_version")) == 0
            )
        if empty_version:
            Table("alembic_version", MetaData(), autoload_with=engine).drop(engine)
            tables.clear()
    if tables:
        engine.dispose()
        pytest.fail("Test database must be empty before migration tests")
    os.environ["RETENTION_DB_TEST_MODE"] = "1"
    config = Config("alembic.ini")
    try:
        command.upgrade(config, "0001_customers_predictions")
        assert "explanations" not in inspect(engine).get_table_names()
        command.upgrade(config, "head")
        inspector = inspect(engine)
        assert {
            "customers",
            "predictions",
            "explanations",
            "recommendations",
            "alembic_version",
        } <= set(inspector.get_table_names())
        with engine.connect() as connection:
            assert connection.scalar(
                text("SELECT version_num FROM alembic_version")
            ) == ("0003_recommendations")
        customer_checks = {
            item["name"] for item in inspector.get_check_constraints("customers")
        }
        prediction_checks = {
            item["name"] for item in inspector.get_check_constraints("predictions")
        }
        assert "ck_customers_gender_category" in customer_checks
        assert "ck_customers_tenure" in customer_checks
        assert "ck_predictions_probability" in prediction_checks
        assert "ck_predictions_class_threshold" in prediction_checks
        assert "ck_predictions_snapshot_object" in prediction_checks
        assert {item["name"] for item in inspector.get_indexes("predictions")} >= {
            "ix_predictions_customer_time",
            "ix_predictions_predicted_at",
        }
        foreign_keys = inspector.get_foreign_keys("predictions")
        assert len(foreign_keys) == 1
        assert foreign_keys[0]["referred_table"] == "customers"
        assert foreign_keys[0]["options"]["ondelete"] == "RESTRICT"
        assert "uq_customers_external_id" in {
            item["name"] for item in inspector.get_unique_constraints("customers")
        }
        explanation_checks = {
            item["name"] for item in inspector.get_check_constraints("explanations")
        }
        assert "ck_explanations_output_space" in explanation_checks
        assert "ck_explanations_probabilities" in explanation_checks
        assert "uq_explanations_request" in {
            item["name"] for item in inspector.get_unique_constraints("explanations")
        }
        explanation_fk = inspector.get_foreign_keys("explanations")
        assert len(explanation_fk) == 1
        assert explanation_fk[0]["referred_table"] == "predictions"
        assert explanation_fk[0]["options"]["ondelete"] == "RESTRICT"
        command.check(config)
        yield make_session_factory(engine)
    finally:
        command.downgrade(config, "0001_customers_predictions")
        assert "explanations" not in inspect(engine).get_table_names()
        command.upgrade(config, "head")
        assert "explanations" in inspect(engine).get_table_names()
        command.downgrade(config, "base")
        assert set(inspect(engine).get_table_names()) == {"alembic_version"}
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT count(*) FROM alembic_version")) == 0
        Table("alembic_version", MetaData(), autoload_with=engine).drop(engine)
        assert not inspect(engine).get_table_names()
        engine.dispose()
        os.environ.pop("RETENTION_DB_TEST_MODE", None)
