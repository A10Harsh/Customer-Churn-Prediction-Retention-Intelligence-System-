"""Ingest raw customer churn data."""

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from retention.utils.config import load_config
from retention.utils.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def read_raw_data(
    input_path: str | Path,
    encoding: str = "utf-8",
    strip_whitespace: bool = True,
) -> pd.DataFrame:
    """Read and normalize the raw CSV dataset."""

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    logger.info("Reading raw dataset: %s", input_path)

    df = pd.read_csv(
        input_path,
        encoding=encoding,
        dtype="string",
        keep_default_na=False,
        on_bad_lines="error",
    )

    if df.empty:
        raise ValueError("The input dataset is empty.")

    if strip_whitespace:
        df = df.apply(lambda column: column.str.strip())

    # Convert empty strings to explicit missing values.
    df = df.replace("", pd.NA)

    logger.info(
        "Ingested %d rows and %d columns",
        df.shape[0],
        df.shape[1],
    )

    return df


def save_ingested_data(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Persist the ingested dataset in Parquet format."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
    )

    logger.info(
        "Ingested dataset saved to %s",
        output_path,
    )


def run_ingestion(
    config_path: str | Path = "params.yaml",
) -> None:
    """Execute the ingestion pipeline stage."""

    config = load_config(config_path)

    data_config = config["data"]
    ingestion_config = config["ingestion"]

    df = read_raw_data(
        input_path=data_config["raw_path"],
        encoding=ingestion_config["encoding"],
        strip_whitespace=ingestion_config["strip_whitespace"],
    )

    save_ingested_data(
        df,
        data_config["ingested_path"],
    )


def main() -> None:
    """Command-line entry point."""

    load_dotenv()
    logger.setLevel(__import__("os").getenv("LOG_LEVEL", "INFO").upper())

    parser = argparse.ArgumentParser(description="Ingest customer churn dataset.")

    parser.add_argument(
        "--config",
        default="params.yaml",
    )

    args = parser.parse_args()

    try:
        run_ingestion(args.config)
    except Exception:
        logger.error("Data ingestion failed.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
