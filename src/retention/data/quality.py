"""Data quality reporting utilities."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def summarize_data(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Generate dataset-level quality statistics."""

    duplicate_ids = (
        int(df["customerID"].duplicated().sum()) if "customerID" in df.columns else None
    )

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": {
            str(column): int(count) for column, count in df.isna().sum().items()
        },
        "total_missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": duplicate_ids,
        "data_types": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
    }


def save_quality_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save the data quality report as JSON."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(".json.tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False,
        )

    temporary_path.replace(output_path)
