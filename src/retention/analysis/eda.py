"""Orchestration and CLI for leakage-safe automated development-data EDA."""

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from retention.analysis.association import (
    categorical_target_associations,
    correlation_ratio_matrix,
    mutual_information_ranking,
    numerical_correlations,
    point_biserial_with_target,
    strong_associations,
)
from retention.analysis.descriptive import (
    categorical_frequencies,
    dataset_quality,
    numerical_statistics,
    rare_categories,
    target_summary,
    variability_screen,
)
from retention.analysis.outliers import outlier_screening
from retention.analysis.reporting import write_html, write_summary, write_tables
from retention.analysis.segmentation import (
    business_kpis,
    pareto_concentration,
    prepare_segments,
    risk_value_segments,
    segment_summary,
    verify_reconciliation,
)
from retention.analysis.visualization import generate_figures
from retention.utils.config import load_config
from retention.utils.logger import get_logger

logger = get_logger(__name__)


class EDAError(RuntimeError):
    """Raised when safe automated EDA generation cannot complete."""


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise EDAError(f"Development dataset is missing columns: {missing}")
    if df.empty:
        raise EDAError("Development dataset is empty.")


def run_eda(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    """Generate and atomically publish all configured EDA artifacts."""
    config = load_config(config_path)
    data_cfg, eda_cfg = config["data"], config["eda"]
    input_path = Path(data_cfg["development_path"])
    output_path = Path(eda_cfg["report_directory"])
    forbidden = {
        Path(data_cfg.get("test_path", "test.parquet")).resolve(),
        Path(data_cfg.get("validated_path", "validated.parquet")).resolve(),
    }
    if input_path.resolve() in forbidden or input_path.name != "development.parquet":
        raise EDAError("EDA input must be the configured development.parquet only.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Development dataset not found: {input_path}")
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_output == resolved_input or resolved_output in resolved_input.parents:
        raise EDAError(
            "Report directory must not be the input or one of its parent directories."
        )
    if resolved_output == Path.cwd().resolve():
        raise EDAError("Report directory must not be the project working directory.")
    numerical = list(eda_cfg["numerical_features"])
    categorical = list(eda_cfg["categorical_features"])
    target, identifier = data_cfg["target_column"], data_cfg["id_column"]
    if identifier in numerical + categorical or target in numerical + categorical:
        raise EDAError("Identifier and target cannot be configured as predictors.")
    df = pd.read_parquet(input_path)
    _validate_columns(df, [identifier, target, *numerical, *categorical])
    logger.info("Loaded development dataset with %d records", len(df))

    quality = dataset_quality(df)
    numerical_stats = numerical_statistics(df, numerical)
    frequencies = categorical_frequencies(df, categorical)
    rare = rare_categories(frequencies, float(eda_cfg["rare_category_threshold"]))
    variability = variability_screen(
        df, numerical + categorical, float(eda_cfg["near_constant_dominance_threshold"])
    )
    target_stats = target_summary(df, target)
    correlations = numerical_correlations(df, numerical)
    point_biserial = point_biserial_with_target(df, numerical, target)
    categorical_assoc = categorical_target_associations(df, categorical, target)
    eta = correlation_ratio_matrix(df, categorical, numerical)
    mi = mutual_information_ranking(
        df,
        numerical,
        categorical,
        target,
        int(config["project"]["random_state"]),
        int(eda_cfg["mutual_information"]["n_neighbors"]),
    )
    thresholds = eda_cfg["strong_association_thresholds"]
    strong = strong_associations(
        correlations,
        categorical_assoc,
        point_biserial,
        float(thresholds["numerical"]),
        float(thresholds["categorical"]),
        float(thresholds["target"]),
    )

    segmented = prepare_segments(df, target, list(eda_cfg["tenure_boundaries"]))
    lifecycle = segment_summary(segmented, "tenure_segment")
    value = segment_summary(segmented, "value_segment")
    contract = segment_summary(segmented, "Contract")
    adoption = segment_summary(segmented, "adoption_band")
    risk_value = risk_value_segments(
        segmented, int(eda_cfg["minimum_segment_sample_size"])
    )
    for table in (lifecycle, value, contract, adoption):
        verify_reconciliation(segmented, table)
    verify_reconciliation(segmented, risk_value)
    churn_pareto = pareto_concentration(risk_value, "churned_count")
    outlier_summary, outlier_tables = outlier_screening(
        df,
        numerical,
        float(eda_cfg["iqr_multiplier"]),
        float(eda_cfg["z_score_threshold"]),
        float(eda_cfg["modified_z_score_threshold"]),
        tuple(eda_cfg["percentile_bounds"]),
    )
    tables = {
        "numerical_statistics": numerical_stats,
        "categorical_frequencies": frequencies,
        "rare_categories": rare,
        "variability_screen": variability,
        **correlations,
        "point_biserial": point_biserial,
        "categorical_target_associations": categorical_assoc,
        "correlation_ratio": eta,
        "mutual_information": mi,
        "strong_associations": strong,
        "lifecycle_segments": lifecycle,
        "monthly_charge_quartiles": value,
        "contract_segments": contract,
        "service_adoption_segments": adoption,
        "contract_lifecycle_risk_value": risk_value,
        "churn_pareto": churn_pareto,
        **{f"outlier_{k}": v for k, v in outlier_tables.items()},
    }
    kpis = business_kpis(segmented)
    summary = {
        "schema_version": "1.0.0",
        "data_source": str(input_path),
        "data_source_sha256": _fingerprint(input_path),
        "dataset_dimensions": {"rows": quality["rows"], "columns": quality["columns"]},
        "data_quality": quality,
        "target_distribution": target_stats,
        "baseline_classification": target_stats["majority_baseline"],
        "numerical_descriptive_statistics": numerical_stats,
        "association_summaries": {
            "point_biserial": point_biserial,
            "categorical_target": categorical_assoc,
            "mutual_information": mi,
            "strong": strong,
        },
        "outlier_screening": outlier_summary,
        "business_kpis": kpis,
        "segmentation_summaries": {
            "lifecycle": lifecycle,
            "contract": contract,
            "service_adoption": adoption,
            "risk_value": risk_value,
        },
        "key_findings": {
            "contract_cramers_v": float(
                categorical_assoc.loc["Contract", "bias_corrected_v"]
            ),
            "tenure_total_charges_pearson": float(
                correlations["pearson"].loc["tenure", "TotalCharges"]
            ),
            "onboarding_churn_rate": float(
                lifecycle.loc["0–6: Onboarding", "churn_rate"]
            ),
        },
        "analytical_configuration": eda_cfg,
        "limitations": [
            "Development split only; findings are descriptive and not hold-out "
            "performance estimates.",
            "MonthlyCharges is an observed exposure proxy, not realized revenue "
            "loss or CLV.",
            "Associations and segments do not establish causality or intervention "
            "effectiveness.",
            "Outlier flags are diagnostic; no observations are removed or transformed.",
            "Isolation Forest and LOF remain notebook-only sensitivity experiments.",
        ],
    }

    staging = output_path.with_name(output_path.name + ".staging")
    backup = output_path.with_name(output_path.name + ".previous")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        table_names = write_tables(tables, staging / "tables")
        figure_names = generate_figures(
            df,
            staging / "figures",
            numerical,
            target,
            correlations["pearson"],
            categorical_assoc,
            mi,
            lifecycle,
            risk_value,
            contract,
            outlier_tables["method_flags"],
            churn_pareto,
            int(eda_cfg["figures"]["dpi"]),
            float(eda_cfg["figures"]["width"]),
            float(eda_cfg["figures"]["height"]),
        )
        write_summary(summary, staging / "summary.json")
        write_html(summary, table_names, figure_names, staging / "report.html")
        shutil.rmtree(backup, ignore_errors=True)
        if output_path.exists():
            output_path.replace(backup)
        staging.replace(output_path)
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not output_path.exists():
            backup.replace(output_path)
        logger.exception(
            "EDA report generation failed; current published report was preserved."
        )
        raise
    logger.info("EDA report published to %s", output_path)
    return summary


def main() -> None:
    """CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate automated development-data EDA."
    )
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    try:
        run_eda(args.config)
    except Exception as exc:
        logger.error("EDA stage terminated: %s", type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
