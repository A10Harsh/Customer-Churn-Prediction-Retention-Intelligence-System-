"""Reproducible and leakage-safe customer data splitting."""

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from retention.data.quality import save_quality_report
from retention.utils.config import load_config
from retention.utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitError(ValueError):
    """Raised when data splitting requirements are violated."""


def validate_split_input(
    df: pd.DataFrame,
    target_column: str,
    id_column: str,
    test_size: float,
) -> None:
    """Validate prerequisites for stratified splitting."""

    if df.empty:
        raise DataSplitError("Cannot split an empty dataset.")

    if target_column == id_column:
        raise DataSplitError("Target and identifier columns must be different.")

    required_columns = {
        target_column,
        id_column,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise DataSplitError(f"Missing required columns: {sorted(missing_columns)}")

    if not 0 < test_size < 1:
        raise DataSplitError("test_size must be between 0 and 1.")

    if df[target_column].isna().any():
        raise DataSplitError("Target column contains missing values.")

    if df[id_column].isna().any():
        raise DataSplitError("Customer ID column contains missing values.")

    if not df[id_column].is_unique:
        raise DataSplitError("Duplicate customer IDs detected.")

    class_counts = df[target_column].value_counts()

    if len(class_counts) != 2:
        raise DataSplitError("Expected exactly two target classes.")

    if class_counts.min() < 2:
        raise DataSplitError("Each target class must contain at least two records.")

    test_count = math.ceil(len(df) * test_size)
    development_count = len(df) - test_count

    if min(test_count, development_count) < len(class_counts):
        raise DataSplitError("Split sizes are too small for all target classes.")

    logger.info(
        "Input validation passed for %d records",
        len(df),
    )


def split_data(
    df: pd.DataFrame,
    target_column: str = "Churn",
    id_column: str = "customerID",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create stratified development and test datasets."""

    validate_split_input(
        df=df,
        target_column=target_column,
        id_column=id_column,
        test_size=test_size,
    )

    # Ensure consistent input ordering before sampling.
    ordered_df = df.sort_values(by=id_column).reset_index(drop=True)

    try:
        development_df, test_df = train_test_split(
            ordered_df,
            test_size=test_size,
            random_state=random_state,
            stratify=ordered_df[target_column],
            shuffle=True,
        )

    except ValueError as exc:
        raise DataSplitError("Stratified splitting failed.") from exc

    development_df = development_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Created development set with %d records",
        len(development_df),
    )

    logger.info(
        "Created test set with %d records",
        len(test_df),
    )

    return development_df, test_df


def verify_split_integrity(
    original_df: pd.DataFrame,
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = "Churn",
    id_column: str = "customerID",
) -> dict[str, Any]:
    """Verify partition integrity and class preservation."""

    if development_df.empty or test_df.empty:
        raise DataSplitError("A generated partition is empty.")

    if len(development_df) + len(test_df) != len(original_df):
        raise DataSplitError("Record count mismatch after splitting.")

    if not development_df[id_column].is_unique or not test_df[id_column].is_unique:
        raise DataSplitError("Duplicate customer IDs found within a partition.")

    original_ids = set(original_df[id_column])
    development_ids = set(development_df[id_column])
    test_ids = set(test_df[id_column])

    overlapping_ids = development_ids.intersection(test_ids)

    if overlapping_ids:
        raise DataSplitError("Customer ID overlap detected between partitions.")

    combined_ids = development_ids.union(test_ids)

    if combined_ids != original_ids:
        raise DataSplitError("Customer records were lost or unexpectedly added.")

    if not (
        development_df.dtypes.equals(original_df.dtypes)
        and test_df.dtypes.equals(original_df.dtypes)
    ):
        raise DataSplitError("Dataset schema changed during splitting.")

    # Verify that splitting did not modify record values.
    reconstructed_df = (
        pd.concat(
            [development_df, test_df],
            ignore_index=True,
        )
        .sort_values(id_column)
        .reset_index(drop=True)
    )

    expected_df = original_df.sort_values(id_column).reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            reconstructed_df,
            expected_df,
            check_exact=True,
        )
    except AssertionError as exc:
        raise DataSplitError("Record values changed during splitting.") from exc

    original_distribution = original_df[target_column].value_counts(normalize=True)

    development_distribution = development_df[target_column].value_counts(
        normalize=True
    )

    test_distribution = test_df[target_column].value_counts(normalize=True)

    if not (
        set(original_distribution.index)
        == set(development_distribution.index)
        == set(test_distribution.index)
    ):
        raise DataSplitError("Target classes are missing from a partition.")

    development_difference = (
        (original_distribution - development_distribution).abs().max()
    )

    test_difference = (original_distribution - test_distribution).abs().max()

    # Allow approximately one record of rounding variation.
    development_tolerance = 1 / len(development_df) + 1e-12

    test_tolerance = 1 / len(test_df) + 1e-12

    if (
        development_difference > development_tolerance
        or test_difference > test_tolerance
    ):
        raise DataSplitError("Target class distribution changed unexpectedly.")

    logger.info("All split integrity checks passed.")

    return {
        "row_count_preserved": True,
        "schema_preserved": True,
        "record_values_preserved": True,
        "overlapping_customer_ids": 0,
        "all_customer_ids_preserved": True,
        "all_target_classes_preserved": True,
        "stratification_verified": True,
        "development_max_class_difference": float(development_difference),
        "test_max_class_difference": float(test_difference),
    }


def summarize_partition(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """Generate aggregate metadata for a dataset partition."""

    counts = df[target_column].value_counts()

    proportions = df[target_column].value_counts(normalize=True)

    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "class_counts": {str(label): int(count) for label, count in counts.items()},
        "class_proportions": {
            str(label): float(proportion) for label, proportion in proportions.items()
        },
    }


def run_splitting(
    config_path: str | Path = "params.yaml",
) -> dict[str, Any]:
    """Execute the configured data splitting stage."""

    config = load_config(config_path)

    data_config = config["data"]
    project_config = config["project"]

    input_path = Path(data_config["validated_path"])

    development_path = Path(data_config["development_path"])

    test_path = Path(data_config["test_path"])

    report_path = Path(data_config["split_report_path"])

    target_column = data_config["target_column"]
    id_column = data_config["id_column"]

    test_size = data_config["test_size"]
    random_state = project_config["random_state"]

    output_paths = [
        development_path,
        test_path,
        report_path,
    ]

    resolved_paths = {path.resolve() for path in [input_path, *output_paths]}

    if len(resolved_paths) != 4:
        raise DataSplitError("Input and output paths must be distinct.")

    if not input_path.is_file():
        raise FileNotFoundError(f"Validated dataset not found: {input_path}")

    df = pd.read_parquet(input_path)

    logger.info(
        "Loaded validated dataset with %d records",
        len(df),
    )

    # Remove stale results from a previous execution.
    for path in output_paths:
        path.unlink(missing_ok=True)

    temporary_paths = [
        development_path.with_suffix(".parquet.tmp"),
        test_path.with_suffix(".parquet.tmp"),
    ]

    try:
        development_df, test_df = split_data(
            df=df,
            target_column=target_column,
            id_column=id_column,
            test_size=test_size,
            random_state=random_state,
        )

        integrity_report = verify_split_integrity(
            original_df=df,
            development_df=development_df,
            test_df=test_df,
            target_column=target_column,
            id_column=id_column,
        )

        report = {
            "status": "passed",
            "split_method": "stratified_random",
            "random_state": random_state,
            "test_size": test_size,
            "target_column": target_column,
            "id_column": id_column,
            "original": summarize_partition(
                df,
                target_column,
            ),
            "development": summarize_partition(
                development_df,
                target_column,
            ),
            "test": summarize_partition(
                test_df,
                target_column,
            ),
            "integrity_checks": integrity_report,
        }

        for path in output_paths:
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        development_df.to_parquet(
            temporary_paths[0],
            engine="pyarrow",
            index=False,
        )

        test_df.to_parquet(
            temporary_paths[1],
            engine="pyarrow",
            index=False,
        )

        temporary_paths[0].replace(development_path)
        temporary_paths[1].replace(test_path)

        save_quality_report(
            report,
            report_path,
        )

    except Exception:
        for path in [
            *output_paths,
            *temporary_paths,
        ]:
            path.unlink(missing_ok=True)

        logger.error("Data splitting failed.")

        raise

    logger.info(
        "Development dataset saved to %s",
        development_path,
    )

    logger.info(
        "Test dataset saved to %s",
        test_path,
    )

    logger.info(
        "Split report saved to %s",
        report_path,
    )

    return report


def main() -> None:
    """Command-line entry point."""

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Create stratified customer data partitions."
    )

    parser.add_argument(
        "--config",
        default="params.yaml",
    )

    args = parser.parse_args()

    try:
        run_splitting(args.config)

    except Exception as exc:
        logger.error(
            "Splitting stage terminated: %s",
            type(exc).__name__,
        )

        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
