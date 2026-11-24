"""Phase 2 source ingestion, validation, splitting, and aggregate analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
import yaml
from pandera import Check
from scipy.stats import chi2_contingency, spearmanr
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
INTERIM = ROOT / "data/interim"
SOURCE_COLUMNS = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
)
CATEGORIES = {
    "gender": ["Female", "Male"],
    "partner": ["Yes", "No"],
    "dependents": ["Yes", "No"],
    "phoneservice": ["Yes", "No"],
    "multiplelines": ["Yes", "No", "No phone service"],
    "internetservice": ["DSL", "Fiber optic", "No"],
    "onlinesecurity": ["Yes", "No", "No internet service"],
    "onlinebackup": ["Yes", "No", "No internet service"],
    "deviceprotection": ["Yes", "No", "No internet service"],
    "techsupport": ["Yes", "No", "No internet service"],
    "streamingtv": ["Yes", "No", "No internet service"],
    "streamingmovies": ["Yes", "No", "No internet service"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperlessbilling": ["Yes", "No"],
    "paymentmethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def config() -> dict[str, Any]:
    """Load and check the non-secret Phase 2 configuration."""
    result = yaml.safe_load((ROOT / "configs/phase2.yaml").read_text())
    if tuple(result["expected_columns"]) != SOURCE_COLUMNS:
        raise ValueError("Source column configuration differs from the contract")
    if abs(sum(result["split"].values()) - 1) > 1e-9:
        raise ValueError("Split ratios must sum to one")
    if result["seed"] != 42:
        raise ValueError("Split seed must be 42")
    return result


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic aggregate JSON without non-finite numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def version(path: Path) -> str:
    """Return a SHA-256 digest for immutable source provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_rows(path: Path) -> list[dict[str, str]]:
    """Read exact CSV strings and reject malformed identity and target fields."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise ValueError("Source header does not match required columns")
        rows = list(reader)
    if not rows or any(
        None in row or any(v is None for v in row.values()) for row in rows
    ):
        raise ValueError("Source is empty or has malformed rows")
    ids = [row["customerID"] for row in rows]
    if any(not identifier.strip() for identifier in ids) or len(ids) != len(set(ids)):
        raise ValueError("Source has blank or duplicate customer IDs")
    if any(row["Churn"] not in {"Yes", "No"} for row in rows):
        raise ValueError("Source has unknown churn labels")
    for row in rows:
        try:
            int(row["SeniorCitizen"])
            int(row["tenure"])
            float(row["MonthlyCharges"])
            if row["TotalCharges"].strip():
                float(row["TotalCharges"])
        except ValueError as exc:
            raise ValueError("Source has invalid numeric fields") from exc
    return rows


def engine() -> Engine:
    """Build a database connection without exposing credentials."""
    names = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise ValueError("Missing database variables: " + ", ".join(missing))
    return create_engine(
        URL.create(
            "postgresql+psycopg",
            username=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            host=os.environ["DB_HOST"],
            port=int(os.environ["DB_PORT"]),
            database=os.environ["DB_NAME"],
        )
    )


def bootstrap() -> None:
    """Create the four schemas and Phase 2 tables without deleting records."""
    with engine().begin() as connection:
        for statement in (ROOT / "database/schema.sql").read_text().split(";"):
            if statement.strip():
                connection.execute(text(statement))


def ingest() -> None:
    """Insert one immutable source version and its staged representation."""
    source = ROOT / config()["source"]
    rows = source_rows(source)
    digest = version(source)
    columns = [name.lower() for name in SOURCE_COLUMNS]
    records = [
        {"source_version": digest, **{key.lower(): value for key, value in row.items()}}
        for row in rows
    ]
    sql = text(
        "INSERT INTO raw.telco_customers (source_version, "
        + ", ".join(columns)
        + ") VALUES (:source_version, "
        + ", ".join(":" + c for c in columns)
        + ")"
    )
    with engine().begin() as connection:
        inserted = connection.execute(
            text(
                "INSERT INTO raw.ingestion_log (source_version, source_rows) "
                "VALUES (:v, :n) ON CONFLICT DO NOTHING RETURNING source_version"
            ),
            {"v": digest, "n": len(rows)},
        ).scalar()
        if inserted:
            connection.execute(sql, records)
            connection.execute(
                text((ROOT / "database/staging.sql").read_text()),
                {"source_version": digest},
            )
        counts = [
            connection.execute(
                text(f"SELECT count(*) FROM {table} WHERE source_version = :v"),
                {"v": digest},
            ).scalar_one()
            for table in ("raw.telco_customers", "staging.telco_customers")
        ]
        if counts != [len(rows), len(rows)]:
            raise ValueError("Source, raw, and staging row counts differ")
    write_json(
        PROCESSED / "ingest_manifest.json",
        {"source_version": digest, "source_rows": len(rows), "staged_rows": counts[1]},
    )


def staged_frame(digest: str) -> pd.DataFrame:
    """Load only rows matching the declared source version."""
    with engine().connect() as connection:
        return pd.read_sql(
            text(
                "SELECT * FROM staging.telco_customers "
                "WHERE source_version = :v ORDER BY customerid"
            ),
            connection,
            params={"v": digest},
        ).drop(columns="source_version")


def staging_schema() -> pa.DataFrameSchema:
    """Return the strict staged data contract."""
    columns: dict[str, pa.Column] = {
        "customerid": pa.Column(str, Check.str_length(min_value=1), unique=True),
        "seniorcitizen": pa.Column(int, Check.isin([0, 1])),
        "tenure": pa.Column(int, Check.ge(0)),
        "monthlycharges": pa.Column(float, Check.ge(0)),
        "totalcharges": pa.Column(float, Check.ge(0), nullable=True),
        "churn": pa.Column(int, Check.isin([0, 1])),
    }
    columns.update(
        {key: pa.Column(str, Check.isin(values)) for key, values in CATEGORIES.items()}
    )
    return pa.DataFrameSchema(columns, strict=True, coerce=True)


def validate() -> None:
    """Validate staging and emit bounded aggregate structural evidence."""
    digest = json.loads((PROCESSED / "ingest_manifest.json").read_text())[
        "source_version"
    ]
    frame = staged_frame(digest)
    try:
        staging_schema().validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        groups = (
            exc.failure_cases.groupby(["column", "check"], dropna=False).size().head(50)
        )
        report = [
            {"column": str(key[0]), "check": str(key[1]), "count": int(count)}
            for key, count in groups.items()
        ]
        write_json(
            PROCESSED / "validation_report.json",
            {"passed": False, "failures": report, "source_version": digest},
        )
        raise ValueError("Staging failed Pandera validation") from None
    write_json(
        PROCESSED / "validation_report.json",
        {"passed": True, "failures": [], "source_version": digest},
    )
    write_json(
        PROCESSED / "structure_profile.json",
        {
            "source_version": digest,
            "scope": "full_dataset_target_unused",
            "rows": len(frame),
            "columns": len(frame.columns),
            "missing": {
                name: int(frame[name].isna().sum())
                for name in frame.columns
                if name not in {"customerid", "churn"}
            },
            "unique_ids": int(frame.customerid.nunique()),
        },
    )


def split() -> None:
    """Create the one fixed stratified split after successful validation."""
    manifest = json.loads((PROCESSED / "validation_report.json").read_text())
    if not manifest["passed"]:
        raise ValueError("Staging validation must pass before splitting")
    digest = manifest["source_version"]
    frame = staged_frame(digest).sort_values("customerid")
    settings = config()
    ratios = settings["split"]
    train, temporary = train_test_split(
        frame,
        test_size=1 - ratios["train"],
        random_state=settings["seed"],
        stratify=frame.churn,
    )
    validation, test = train_test_split(
        temporary,
        test_size=ratios["test"] / (ratios["test"] + ratios["validation"]),
        random_state=settings["seed"],
        stratify=temporary.churn,
    )
    parts = {"train": train, "validation": validation, "test": test}
    ids = [set(part.customerid) for part in parts.values()]
    if sum(map(len, ids)) != len(set.union(*ids)) or sum(map(len, ids)) != len(frame):
        raise ValueError("Split overlap or incomplete coverage")
    INTERIM.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.sort_values("customerid").to_csv(
            INTERIM / f"{name}.csv", index=False, float_format="%.8f"
        )
    write_json(
        PROCESSED / "split_manifest.json",
        {
            "source_version": digest,
            "seed": settings["seed"],
            "ratios": settings["split"],
            "sizes": {name: len(part) for name, part in parts.items()},
            "churn_rates": {
                name: round(float(part.churn.mean()), 8) for name, part in parts.items()
            },
        },
    )


def cramers_v(table: pd.DataFrame) -> float:
    """Compute bias-corrected Cramer's V for an aggregate contingency table."""
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi = max(0.0, chi2 / n - (table.shape[0] - 1) * (table.shape[1] - 1) / (n - 1))
    rows = table.shape[0] - (table.shape[0] - 1) ** 2 / (n - 1)
    cols = table.shape[1] - (table.shape[1] - 1) ** 2 / (n - 1)
    return round(float((phi / min(rows - 1, cols - 1)) ** 0.5), 6)


def analyze() -> None:
    """Analyze only training rows and render an aggregate Markdown report."""
    structure = json.loads((PROCESSED / "structure_profile.json").read_text())
    frame = pd.read_csv(INTERIM / "train.csv")
    digest = structure["source_version"]
    settings = config()["analysis"]
    metrics: dict[str, Any] = {
        "source_version": digest,
        "scope": "training_only",
        "training_rows": len(frame),
        "target_counts": {
            "no": int((frame.churn == 0).sum()),
            "yes": int((frame.churn == 1).sum()),
        },
        "churn_rate": round(float(frame.churn.mean()), 6),
        "numeric": {},
        "categorical": {},
    }
    for name in ("seniorcitizen", "tenure", "monthlycharges", "totalcharges"):
        values = frame[name]
        pair = frame[[name, "churn"]].dropna()
        corr = pair[name].corr(pair.churn)
        rank = spearmanr(pair[name], pair.churn).statistic
        by_target = pair.groupby("churn")[name].agg(["size", "mean", "median"])
        metrics["numeric"][name] = {
            "missing": int(values.isna().sum()),
            "non_missing": int(values.notna().sum()),
            "min": round(float(values.min()), 6),
            "p25": round(float(values.quantile(0.25)), 6),
            "mean": round(float(values.mean()), 6),
            "median": round(float(values.median()), 6),
            "p75": round(float(values.quantile(0.75)), 6),
            "max": round(float(values.max()), 6),
            "correlation": None if pd.isna(corr) else round(float(corr), 6),
            "spearman_rho": None if pd.isna(rank) else round(float(rank), 6),
            "pairwise_n": len(pair),
            "by_target": {
                str(label): {
                    "count": int(row["size"]),
                    "mean": round(float(row["mean"]), 6),
                    "median": round(float(row["median"]), 6),
                }
                for label, row in by_target.iterrows()
            },
            "proxy_flag": bool(abs(corr) > settings["correlation_flag"]),
        }
    for name in CATEGORIES:
        groups = frame.groupby(name, dropna=False).churn.agg(["size", "mean"])
        table = pd.crosstab(frame[name], frame.churn)
        association = cramers_v(table)
        metrics["categorical"][name] = {
            "levels": {
                str(level): {
                    "count": int(row["size"]),
                    "churn_rate": round(float(row["mean"]), 6),
                    "sparse": bool(row["size"] < settings["sparse_level_min"]),
                }
                for level, row in groups.iterrows()
            },
            "cramers_v": association,
            "proxy_flag": association > settings["cramers_v_flag"],
        }
    metrics["blank_totalcharges_zero_tenure"] = int(
        ((frame.totalcharges.isna()) & (frame.tenure == 0)).sum()
    )
    metrics["totalcharges_missing"] = int(frame.totalcharges.isna().sum())
    write_json(PROCESSED / "analysis_metrics.json", metrics)
    lines = [
        "# Data analysis report",
        "",
        f"Source SHA-256: `{digest}`",
        "",
        "Generated by `.venv/bin/python -m src.churn_phase2 analyze`.",
        "",
        "## Full dataset structure (target unused)",
        "",
        f"Rows: {structure['rows']}; columns: {structure['columns']}; "
        f"unique IDs: {structure['unique_ids']}.",
        "",
        "Missing values by field (nonzero only):",
        "",
    ]
    missing_fields = {
        name: count for name, count in structure["missing"].items() if count
    }
    lines.extend(
        [f"- `{name}`: {count}" for name, count in missing_fields.items()] or ["- None"]
    )
    lines.extend(
        [
            "",
            "## Training partition only",
            "",
            f"Rows: {len(frame)}; churn counts: "
            f"No={metrics['target_counts']['no']}, "
            f"Yes={metrics['target_counts']['yes']}; "
            f"churn rate: {metrics['churn_rate']:.6f}.",
            "",
            "| Numeric field | Available | Missing | Min | P25 | Median | P75 | "
            "Max | Pearson r | Spearman rho |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, item in metrics["numeric"].items():
        lines.append(
            f"| {name} | {item['pairwise_n']} | {item['missing']} | "
            f"{item['min']} | {item['p25']} | {item['median']} | "
            f"{item['p75']} | {item['max']} | {item['correlation']} | "
            f"{item['spearman_rho']} |"
        )
    lines.extend(
        [
            "",
            "| Numeric field | Non-churn mean | Churn mean |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, item in metrics["numeric"].items():
        lines.append(
            f"| {name} | {item['by_target']['0']['mean']} | "
            f"{item['by_target']['1']['mean']} |"
        )
    lines.extend(
        [
            "",
            f"Missing TotalCharges: {metrics['totalcharges_missing']}; "
            "of these, zero-tenure rows: "
            f"{metrics['blank_totalcharges_zero_tenure']}.",
        ]
    )
    lines.extend(
        [
            "",
            "| Category field | Level | Count | Churn rate | Sparse |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for name, item in metrics["categorical"].items():
        for level, values in item["levels"].items():
            lines.append(
                f"| {name} | {level} | {values['count']} | "
                f"{values['churn_rate']} | {values['sparse']} |"
            )
    lines.extend(
        [
            "",
            "| Categorical field | Corrected Cramer's V | Proxy review flag |",
            "| --- | ---: | --- |",
        ]
    )
    for name, item in metrics["categorical"].items():
        lines.append(f"| {name} | {item['cramers_v']} | {item['proxy_flag']} |")
    lines.extend(
        [
            "",
            "Associations are exploratory and non-causal. "
            "Multiple comparisons and sparse categories limit interpretation. "
            "Proxy flags require Phase 3 review. The source has no event date "
            "or label observation window, so temporal leakage and censoring "
            "cannot be assessed from this file.",
            "",
        ]
    )
    path = ROOT / "docs/data-analysis-report.md"
    path.write_text("\n".join(lines))


def main() -> None:
    """Run one phase stage or the complete pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("bootstrap", "ingest", "validate", "split", "analyze", "all")
    )
    stage = parser.parse_args().stage
    tasks = {
        "bootstrap": bootstrap,
        "ingest": ingest,
        "validate": validate,
        "split": split,
        "analyze": analyze,
    }
    for name in tasks if stage == "all" else (stage,):
        tasks[name]()


if __name__ == "__main__":
    main()
