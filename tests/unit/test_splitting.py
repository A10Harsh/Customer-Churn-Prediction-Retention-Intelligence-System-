"""Unit tests for reproducible, leakage-safe data splitting."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from retention.data.splitting import (
    DataSplitError,
    run_splitting,
    split_data,
    verify_split_integrity,
)


@pytest.fixture
def customer_df() -> pd.DataFrame:
    """Create a balanced dataset large enough for stratification."""

    return pd.DataFrame(
        {
            "customerID": [f"CUST-{index:03d}" for index in range(20)],
            "Churn": ["No", "Yes"] * 10,
            "tenure": list(range(20)),
        }
    )


def test_split_data_is_reproducible_and_stratified(
    customer_df: pd.DataFrame,
) -> None:
    first_development, first_test = split_data(customer_df)
    second_development, second_test = split_data(customer_df.sample(frac=1))

    pd.testing.assert_frame_equal(first_development, second_development)
    pd.testing.assert_frame_equal(first_test, second_test)
    assert first_development.shape == (16, 3)
    assert first_test.shape == (4, 3)
    assert first_test["Churn"].value_counts().to_dict() == {"No": 2, "Yes": 2}


@pytest.mark.parametrize(
    ("dataframe", "message"),
    [
        (pd.DataFrame(), "empty dataset"),
        (
            pd.DataFrame({"customerID": ["1", "2"], "Churn": ["No", None]}),
            "missing values",
        ),
        (
            pd.DataFrame({"customerID": ["1", "1"], "Churn": ["No", "Yes"]}),
            "Duplicate customer IDs",
        ),
        (
            pd.DataFrame({"customerID": ["1", "2"], "Churn": ["No", "No"]}),
            "exactly two target classes",
        ),
    ],
)
def test_split_data_rejects_invalid_input(
    dataframe: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(DataSplitError, match=message):
        split_data(dataframe)


def test_verify_split_integrity_detects_overlap(
    customer_df: pd.DataFrame,
) -> None:
    development, test = split_data(customer_df)
    test.loc[0, "customerID"] = development.loc[0, "customerID"]

    with pytest.raises(DataSplitError, match="overlap"):
        verify_split_integrity(customer_df, development, test)


def test_run_splitting_writes_partitions_and_report(
    tmp_path: Path,
    customer_df: pd.DataFrame,
) -> None:
    input_path = tmp_path / "validated.parquet"
    development_path = tmp_path / "development.parquet"
    test_path = tmp_path / "test.parquet"
    report_path = tmp_path / "split-report.json"
    config_path = tmp_path / "params.yaml"

    customer_df.to_parquet(input_path, index=False)
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"random_state": 42},
                "data": {
                    "validated_path": str(input_path),
                    "development_path": str(development_path),
                    "test_path": str(test_path),
                    "split_report_path": str(report_path),
                    "target_column": "Churn",
                    "id_column": "customerID",
                    "test_size": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )

    report = run_splitting(config_path)

    assert report["status"] == "passed"
    assert report["development"]["row_count"] == 16
    assert report["test"]["row_count"] == 4
    assert development_path.is_file()
    assert test_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_run_splitting_rejects_reused_paths(
    tmp_path: Path,
) -> None:
    shared_path = tmp_path / "shared.parquet"
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"random_state": 42},
                "data": {
                    "validated_path": str(shared_path),
                    "development_path": str(shared_path),
                    "test_path": str(tmp_path / "test.parquet"),
                    "split_report_path": str(tmp_path / "report.json"),
                    "target_column": "Churn",
                    "id_column": "customerID",
                    "test_size": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataSplitError, match="paths must be distinct"):
        run_splitting(config_path)
