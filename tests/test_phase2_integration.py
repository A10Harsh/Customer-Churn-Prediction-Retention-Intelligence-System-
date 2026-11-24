"""PostgreSQL smoke test for immutable, versioned Phase 2 ingestion."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import text
from src import churn_phase2 as phase2

pytestmark = pytest.mark.integration


def test_real_source_ingestion_is_idempotent() -> None:
    """Bootstrap and load twice, preserving raw and staged counts."""
    phase2.bootstrap()
    phase2.ingest()
    first = json.loads((phase2.PROCESSED / "ingest_manifest.json").read_text())
    phase2.ingest()
    second = json.loads((phase2.PROCESSED / "ingest_manifest.json").read_text())
    assert first == second
    with phase2.engine().connect() as connection:
        for table in ("raw.telco_customers", "staging.telco_customers"):
            count = connection.execute(
                text(f"SELECT count(*) FROM {table} WHERE source_version = :v"),
                {"v": first["source_version"]},
            ).scalar_one()
            assert count == first["source_rows"]
    phase2.validate()
