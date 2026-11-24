"""Core Phase 2 contract tests using synthetic, non-identifying data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest
from src import churn_phase2 as phase2


def test_source_contract_rejects_duplicate_and_unknown_target(tmp_path: Path) -> None:
    rows = [dict.fromkeys(phase2.SOURCE_COLUMNS, "No") for _ in range(2)]
    for index, row in enumerate(rows):
        row.update(
            customerID=f"fake-{index}",
            Churn="No",
            SeniorCitizen="0",
            tenure="1",
            MonthlyCharges="10",
            TotalCharges="10",
        )
    path = tmp_path / "sample.csv"

    def save() -> None:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=phase2.SOURCE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    save()
    assert len(phase2.source_rows(path)) == 2
    rows[1]["customerID"] = rows[0]["customerID"]
    save()
    with pytest.raises(ValueError, match="duplicate"):
        phase2.source_rows(path)
    rows[1]["customerID"] = "fake-1"
    rows[1]["Churn"] = "Maybe"
    save()
    with pytest.raises(ValueError, match="unknown churn"):
        phase2.source_rows(path)


def test_staging_schema_accepts_nullable_charges_and_rejects_bad_category() -> None:
    data = {name: [values[0]] for name, values in phase2.CATEGORIES.items()}
    data.update(
        customerid=["fake-1"],
        seniorcitizen=[0],
        tenure=[0],
        monthlycharges=[10.0],
        totalcharges=[None],
        churn=[1],
    )
    frame = pd.DataFrame(data)
    assert phase2.staging_schema().validate(frame).totalcharges.isna().all()
    frame.loc[0, "contract"] = "invalid"
    with pytest.raises(pa.errors.SchemaErrors):
        phase2.staging_schema().validate(frame, lazy=True)


def test_config_has_fixed_split() -> None:
    settings = phase2.config()
    assert settings["seed"] == 42
    assert settings["split"] == {"train": 0.70, "validation": 0.15, "test": 0.15}
    assert "customerID" not in phase2.CATEGORIES
    assert "Churn" not in phase2.CATEGORIES


def test_invalid_category_blocks_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = {name: [values[0]] for name, values in phase2.CATEGORIES.items()}
    data.update(
        customerid=["fake-1"],
        seniorcitizen=[0],
        tenure=[0],
        monthlycharges=[10.0],
        totalcharges=[None],
        churn=[1],
    )
    frame = pd.DataFrame(data)
    frame.loc[0, "contract"] = "invalid"
    monkeypatch.setattr(phase2, "PROCESSED", tmp_path)
    monkeypatch.setattr(phase2, "staged_frame", lambda _digest: frame)
    (tmp_path / "ingest_manifest.json").write_text('{"source_version": "fake"}')
    with pytest.raises(ValueError, match="Pandera"):
        phase2.validate()
    report = json.loads((tmp_path / "validation_report.json").read_text())
    assert report["passed"] is False
    assert report["failures"][0]["column"] == "contract"
    assert "fake-1" not in (tmp_path / "validation_report.json").read_text()
    with pytest.raises(ValueError, match="must pass"):
        phase2.split()
