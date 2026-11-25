"""Unit tests for data ingestion."""

from pathlib import Path

import pandas as pd
import pytest

from retention.data.ingestion import (
    read_raw_data,
    save_ingested_data,
)


def test_read_valid_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "test.csv"

    input_path.write_text(
        "customerID,tenure\nTEST-001,12\n",
        encoding="utf-8",
    )

    df = read_raw_data(input_path)

    assert df.shape == (1, 2)
    assert df.loc[0, "customerID"] == "TEST-001"


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_raw_data(tmp_path / "missing.csv")


def test_blank_values(tmp_path: Path) -> None:
    input_path = tmp_path / "test.csv"

    input_path.write_text(
        "customerID,TotalCharges\nTEST-001,   \n",
        encoding="utf-8",
    )

    df = read_raw_data(input_path)

    assert pd.isna(df.loc[0, "TotalCharges"])


def test_strip_whitespace(tmp_path: Path) -> None:
    input_path = tmp_path / "test.csv"

    input_path.write_text(
        "customerID,Contract\nTEST-001, One year \n",
        encoding="utf-8",
    )

    df = read_raw_data(input_path)

    assert df.loc[0, "Contract"] == "One year"


def test_empty_dataset(tmp_path: Path) -> None:
    input_path = tmp_path / "test.csv"

    input_path.write_text(
        "customerID,tenure\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        read_raw_data(input_path)


def test_save_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({"customerID": ["TEST-001"]})

    output_path = tmp_path / "data.parquet"

    save_ingested_data(df, output_path)

    loaded_df = pd.read_parquet(output_path)

    pd.testing.assert_frame_equal(
        df,
        loaded_df,
    )
