"""Validate and standardize the customer churn dataset."""

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandera.errors import SchemaError, SchemaErrors

from retention.data.quality import (
    save_quality_report,
    summarize_data,
)
from retention.data.schemas import TELCO_SCHEMA
from retention.utils.config import load_config
from retention.utils.logger import get_logger

logger = get_logger(__name__)

INTEGER_COLUMNS = [
    "SeniorCitizen",
    "tenure",
]

FLOAT_COLUMNS = [
    "MonthlyCharges",
    "TotalCharges",
]


class DataValidationError(ValueError):
    """Raised when dataset quality requirements are violated."""


def convert_numeric_column(
    series: pd.Series,
) -> pd.Series:
    """Convert numeric values without silently hiding errors."""

    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid = series.notna() & converted.isna()

    if invalid.any():
        raise DataValidationError(
            f"{series.name}: {int(invalid.sum())} invalid numeric values."
        )

    return converted


def prepare_data(
    df: pd.DataFrame,
    fill_zero_tenure_total_charges: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply deterministic cleaning and type conversion."""

    df = df.copy()

    expected_columns = set(TELCO_SCHEMA.columns)
    actual_columns = set(df.columns)

    missing_columns = expected_columns - actual_columns
    unexpected_columns = actual_columns - expected_columns

    if missing_columns or unexpected_columns:
        raise DataValidationError(
            f"Missing columns: {sorted(missing_columns)}; "
            f"Unexpected columns: {sorted(unexpected_columns)}"
        )

    if not df.columns.is_unique:
        raise DataValidationError("Duplicate column names detected.")

    corrections = {
        "zero_tenure_total_charges_filled": 0,
    }

    for column in INTEGER_COLUMNS:
        numeric = convert_numeric_column(df[column])

        if numeric.isna().any():
            raise DataValidationError(f"{column} contains missing values.")

        if not np.isfinite(numeric).all():
            raise DataValidationError(f"{column} contains non-finite values.")

        if (numeric % 1 != 0).any():
            raise DataValidationError(f"{column} contains non-integer values.")

        try:
            df[column] = numeric.astype("int64")
        except (ValueError, OverflowError, TypeError):
            raise DataValidationError(f"{column} cannot be safely converted.") from None

    for column in FLOAT_COLUMNS:
        df[column] = convert_numeric_column(df[column]).astype("float64")

    missing_charges = df["TotalCharges"].isna()

    eligible_for_correction = missing_charges & (df["tenure"] == 0)

    if fill_zero_tenure_total_charges:
        correction_count = int(eligible_for_correction.sum())

        df.loc[
            eligible_for_correction,
            "TotalCharges",
        ] = 0.0

        corrections["zero_tenure_total_charges_filled"] = correction_count

        logger.info(
            "Corrected %d zero-tenure TotalCharges values",
            correction_count,
        )

    # Standardize all remaining columns as strings.
    for column in expected_columns - set(INTEGER_COLUMNS + FLOAT_COLUMNS):
        df[column] = df[column].astype("string")

    return df, corrections


def validate_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate all schema and business requirements."""

    return TELCO_SCHEMA.validate(
        df,
        lazy=True,
    )


def run_validation(
    config_path: str | Path = "params.yaml",
) -> dict[str, Any]:
    """Execute the validation stage and generate its report."""

    config = load_config(config_path)

    data_config = config["data"]
    validation_config = config["validation"]

    input_path = Path(data_config["ingested_path"])
    output_path = Path(data_config["validated_path"])
    report_path = Path(data_config["quality_report_path"])

    # Remove stale outputs from any earlier execution.
    output_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)

    report: dict[str, Any] = {
        "schema_name": TELCO_SCHEMA.name,
        "status": "failed",
        "before_validation": None,
        "after_validation": None,
        "corrections": {},
        "validation_errors": [],
    }

    temporary_path = output_path.with_suffix(".parquet.tmp")

    try:
        df = pd.read_parquet(input_path)

        report["before_validation"] = summarize_data(df)

        cleaned_df, corrections = prepare_data(
            df,
            fill_zero_tenure_total_charges=validation_config[
                "fill_zero_tenure_total_charges"
            ],
        )

        report["corrections"] = corrections

        validated_df = validate_data(cleaned_df)

        report["after_validation"] = summarize_data(validated_df)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        validated_df.to_parquet(
            temporary_path,
            engine="pyarrow",
            index=False,
        )

        temporary_path.replace(output_path)

        report["status"] = "passed"

        save_quality_report(report, report_path)

        logger.info(
            "Validation passed: %d rows",
            len(validated_df),
        )

        return report

    except (SchemaErrors, SchemaError) as exc:
        failure_cases = exc.failure_cases

        if isinstance(failure_cases, pd.DataFrame):
            report["validation_errors"] = [
                {
                    "column": str(row.get("column")),
                    "check": str(row.get("check")),
                    "index": str(row.get("index")),
                }
                for row in failure_cases.head(100).to_dict(orient="records")
            ]

            report["failure_case_count"] = int(len(failure_cases))
        else:
            report["validation_errors"] = [{"error": "Schema validation failed."}]

    except DataValidationError as exc:
        report["validation_errors"] = [{"error": str(exc)}]

    except Exception as exc:
        report["validation_errors"] = [{"error_type": type(exc).__name__}]

    # Never expose stale or partially validated data.
    output_path.unlink(missing_ok=True)
    temporary_path.unlink(missing_ok=True)

    report["status"] = "failed"

    save_quality_report(report, report_path)

    logger.error(
        "Validation failed. Inspect %s",
        report_path,
    )

    raise DataValidationError("Dataset failed validation.")


def main() -> None:
    """Command-line entry point."""

    load_dotenv()

    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

    parser = argparse.ArgumentParser(description="Validate customer churn dataset.")

    parser.add_argument(
        "--config",
        default="params.yaml",
    )

    args = parser.parse_args()

    try:
        run_validation(args.config)
    except Exception:
        logger.error("Validation stage terminated.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
