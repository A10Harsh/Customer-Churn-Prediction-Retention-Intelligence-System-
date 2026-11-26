"""Integration tests for the complete data splitting pipeline."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from retention.data.ingestion import run_ingestion
from retention.data.splitting import (
    DataSplitError,
    run_splitting,
)
from retention.data.validation import run_validation


@pytest.fixture
def synthetic_customer_dataset(
    valid_customer_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate 100 schema-compatible customer records."""

    df = pd.concat(
        [valid_customer_df] * 100,
        ignore_index=True,
    )

    df["customerID"] = [f"TEST-{i:04d}" for i in range(100)]

    df["Churn"] = ["No"] * 75 + ["Yes"] * 25

    return df


@pytest.fixture
def split_pipeline_config(
    tmp_path: Path,
) -> Path:
    """Create an isolated pipeline configuration."""

    config = {
        "project": {
            "random_state": 42,
        },
        "data": {
            "raw_path": str(tmp_path / "raw.csv"),
            "ingested_path": str(tmp_path / "ingested.parquet"),
            "validated_path": str(tmp_path / "validated.parquet"),
            "development_path": str(tmp_path / "development.parquet"),
            "test_path": str(tmp_path / "test.parquet"),
            "quality_report_path": str(tmp_path / "quality.json"),
            "split_report_path": str(tmp_path / "split_report.json"),
            "target_column": "Churn",
            "id_column": "customerID",
            "test_size": 0.20,
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


def test_complete_data_pipeline(
    tmp_path: Path,
    split_pipeline_config: Path,
    synthetic_customer_dataset: pd.DataFrame,
) -> None:
    """Verify ingestion, validation and splitting together."""

    synthetic_customer_dataset.to_csv(
        tmp_path / "raw.csv",
        index=False,
    )

    run_ingestion(split_pipeline_config)

    validation_report = run_validation(split_pipeline_config)

    split_report = run_splitting(split_pipeline_config)

    assert validation_report["status"] == "passed"
    assert split_report["status"] == "passed"

    development = pd.read_parquet(tmp_path / "development.parquet")

    test = pd.read_parquet(tmp_path / "test.parquet")

    assert development.shape == (80, 21)
    assert test.shape == (20, 21)

    assert (development["Churn"] == "Yes").sum() == 20
    assert (test["Churn"] == "Yes").sum() == 5

    assert set(development["customerID"]).isdisjoint(set(test["customerID"]))

    assert development.isna().sum().sum() == 0

    assert test.isna().sum().sum() == 0

    assert split_report["integrity_checks"]["overlapping_customer_ids"] == 0

    saved_report = json.loads(
        (tmp_path / "split_report.json").read_text(encoding="utf-8")
    )

    assert saved_report == split_report


def test_pipeline_reproducibility(
    tmp_path: Path,
    split_pipeline_config: Path,
    synthetic_customer_dataset: pd.DataFrame,
) -> None:
    """Verify identical results across repeated executions."""

    synthetic_customer_dataset.to_csv(
        tmp_path / "raw.csv",
        index=False,
    )

    run_ingestion(split_pipeline_config)
    run_validation(split_pipeline_config)

    report_1 = run_splitting(split_pipeline_config)

    development_1 = pd.read_parquet(tmp_path / "development.parquet")

    test_1 = pd.read_parquet(tmp_path / "test.parquet")

    report_2 = run_splitting(split_pipeline_config)

    development_2 = pd.read_parquet(tmp_path / "development.parquet")

    test_2 = pd.read_parquet(tmp_path / "test.parquet")

    pd.testing.assert_frame_equal(
        development_1,
        development_2,
        check_exact=True,
    )

    pd.testing.assert_frame_equal(
        test_1,
        test_2,
        check_exact=True,
    )

    assert report_1 == report_2


def test_pipeline_rejects_invalid_split(
    tmp_path: Path,
    split_pipeline_config: Path,
    synthetic_customer_dataset: pd.DataFrame,
) -> None:
    """Ensure failed splitting removes stale outputs."""

    synthetic_customer_dataset.to_csv(
        tmp_path / "raw.csv",
        index=False,
    )

    run_ingestion(split_pipeline_config)
    run_validation(split_pipeline_config)

    # First create valid pipeline outputs.
    run_splitting(split_pipeline_config)

    assert (tmp_path / "development.parquet").exists()

    assert (tmp_path / "test.parquet").exists()

    # Simulate an upstream dataset containing
    # only a single target class.
    synthetic_customer_dataset["Churn"] = "No"

    synthetic_customer_dataset.to_csv(
        tmp_path / "raw.csv",
        index=False,
    )

    run_ingestion(split_pipeline_config)
    run_validation(split_pipeline_config)

    with pytest.raises(DataSplitError):
        run_splitting(split_pipeline_config)

    assert not (tmp_path / "development.parquet").exists()

    assert not (tmp_path / "test.parquet").exists()

    assert not (tmp_path / "split_report.json").exists()
