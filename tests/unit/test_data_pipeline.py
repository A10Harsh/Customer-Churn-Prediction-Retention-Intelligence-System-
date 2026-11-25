"""Integration tests for the data pipeline."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from retention.data.ingestion import run_ingestion
from retention.data.validation import (
    DataValidationError,
    run_validation,
)


@pytest.fixture
def pipeline_config(tmp_path: Path) -> Path:
    """Create an isolated pipeline configuration."""

    config = {
        "data": {
            "raw_path": str(tmp_path / "raw.csv"),
            "ingested_path": str(tmp_path / "ingested.parquet"),
            "validated_path": str(tmp_path / "validated.parquet"),
            "quality_report_path": str(tmp_path / "quality.json"),
        },
        "ingestion": {
            "encoding": "utf-8",
            "strip_whitespace": True,
        },
        "validation": {
            "fill_zero_tenure_total_charges": True,
        },
    }

    config_path = tmp_path / "params.yaml"

    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    return config_path


def test_successful_pipeline(
    tmp_path: Path,
    pipeline_config: Path,
    valid_customer_df: pd.DataFrame,
) -> None:
    valid_customer_df.to_csv(
        tmp_path / "raw.csv",
        index=False,
    )

    run_ingestion(pipeline_config)
    report = run_validation(pipeline_config)

    assert report["status"] == "passed"

    output = tmp_path / "validated.parquet"

    assert output.is_file()

    df = pd.read_parquet(output)

    assert df.shape == (1, 21)


def test_failed_pipeline(
    tmp_path: Path,
    pipeline_config: Path,
    valid_customer_df: pd.DataFrame,
) -> None:
    raw_path = tmp_path / "raw.csv"

    # First produce a successful pipeline output.
    valid_customer_df.to_csv(raw_path, index=False)

    run_ingestion(pipeline_config)
    run_validation(pipeline_config)

    assert (tmp_path / "validated.parquet").is_file()

    # Introduce an invalid target and rerun.
    valid_customer_df.loc[0, "Churn"] = "Invalid"

    valid_customer_df.to_csv(raw_path, index=False)

    run_ingestion(pipeline_config)

    with pytest.raises(DataValidationError):
        run_validation(pipeline_config)

    # A failed pipeline must not expose stale data.
    assert not (tmp_path / "validated.parquet").exists()

    report = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))

    assert report["status"] == "failed"
    assert report["validation_errors"]
