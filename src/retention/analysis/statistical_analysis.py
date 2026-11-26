"""Orchestration and CLI for leakage-safe automated statistical analysis."""

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from retention.analysis.inference import (
    categorical_independence_test,
    mann_whitney_test,
    welch_mean_test,
    wilson_interval,
)
from retention.analysis.multicollinearity import (
    calculate_vif,
    matrix_diagnostics,
    reference_coded_design,
)
from retention.analysis.multiplicity import adjust_hypothesis_families
from retention.analysis.reporting import (
    write_statistical_html,
    write_summary,
    write_tables,
)
from retention.utils.config import load_config
from retention.utils.logger import get_logger

logger = get_logger(__name__)


class StatisticalAnalysisError(RuntimeError):
    """Raised when safe automated statistical analysis cannot complete."""


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_paths(input_path: Path, output_path: Path, data_config: dict) -> None:
    forbidden = {
        Path(data_config.get("test_path", "test.parquet")).resolve(),
        Path(data_config.get("validated_path", "validated.parquet")).resolve(),
    }
    if input_path.resolve() in forbidden or input_path.name != "development.parquet":
        raise StatisticalAnalysisError(
            "Statistical analysis input must be configured development.parquet only."
        )
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_output == resolved_input or resolved_output in resolved_input.parents:
        raise StatisticalAnalysisError(
            "Output directory must not be the input or one of its parent directories."
        )
    if resolved_output == Path.cwd().resolve():
        raise StatisticalAnalysisError(
            "Output directory must not be the project working directory."
        )


def _validate_columns(data: pd.DataFrame, required: list[str]) -> None:
    if data.empty:
        raise StatisticalAnalysisError("Development dataset is empty.")
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise StatisticalAnalysisError(f"Development dataset is missing: {missing}")
    if data[required].isna().any().any():
        raise StatisticalAnalysisError(
            "Configured analysis columns contain missing values."
        )


def _interpret_vif(value: float, thresholds: list[float]) -> str:
    low, high = sorted(float(item) for item in thresholds)
    if not np.isfinite(value):
        return "Exact dependency; coefficient is not separately identifiable."
    if value >= high:
        return "High finite inflation under this encoding; test stability."
    if value >= low:
        return "Elevated finite inflation; review with domain context."
    return "Below configured screening thresholds."


def _confidence_intervals(
    data: pd.DataFrame,
    target: str,
    numerical_tests: pd.DataFrame,
    segment_features: list[str],
    confidence_level: float,
    minimum_segment_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positive = data[target].eq("Yes")
    lower, upper = wilson_interval(int(positive.sum()), len(data), confidence_level)
    rows.append(
        {
            "interval_id": "overall_churn_rate",
            "estimand": "Churn proportion",
            "group": "All development customers",
            "sample_size": len(data),
            "successes": int(positive.sum()),
            "estimate": float(positive.mean()),
            "confidence_level": confidence_level,
            "ci_lower": lower,
            "ci_upper": upper,
            "method": "Wilson score interval",
        }
    )
    for row in numerical_tests.itertuples(index=False):
        rows.append(
            {
                "interval_id": f"{row.hypothesis_id}_mean_difference",
                "estimand": "Mean difference (churned minus retained)",
                "group": row.predictor,
                "sample_size": int(row.churned_n + row.retained_n),
                "successes": pd.NA,
                "estimate": row.mean_difference,
                "confidence_level": confidence_level,
                "ci_lower": row.ci_lower,
                "ci_upper": row.ci_upper,
                "method": "Welch-Satterthwaite interval",
            }
        )
    for feature in segment_features:
        for level, segment in data.groupby(feature, observed=True, sort=True):
            if len(segment) < minimum_segment_size:
                continue
            successes = int(segment[target].eq("Yes").sum())
            lower, upper = wilson_interval(successes, len(segment), confidence_level)
            rows.append(
                {
                    "interval_id": f"{feature}={level}",
                    "estimand": "Segment churn proportion",
                    "group": f"{feature}={level}",
                    "sample_size": len(segment),
                    "successes": successes,
                    "estimate": successes / len(segment),
                    "confidence_level": confidence_level,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "method": "Wilson score interval",
                }
            )
    return pd.DataFrame(rows)


def _numerical_tests(
    data: pd.DataFrame,
    target: str,
    hypotheses: dict[str, str],
    confidence_level: float,
) -> pd.DataFrame:
    rows = []
    for hypothesis_id, predictor in hypotheses.items():
        churned = data.loc[data[target].eq("Yes"), predictor]
        retained = data.loc[data[target].eq("No"), predictor]
        welch = welch_mean_test(churned, retained, confidence_level)
        rank = mann_whitney_test(churned, retained)
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "predictor": predictor,
                "direction": "churned_minus_retained",
                "churned_n": welch["group_one_n"],
                "retained_n": welch["group_two_n"],
                "churned_mean": welch["group_one_mean"],
                "retained_mean": welch["group_two_mean"],
                "mean_difference": welch["mean_difference"],
                "welch_t": welch["test_statistic"],
                "welch_df": welch["degrees_of_freedom"],
                "welch_raw_p_value": welch["raw_p_value"],
                "ci_lower": welch["ci_lower"],
                "ci_upper": welch["ci_upper"],
                "cohens_d": welch["cohens_d"],
                "mann_whitney_u": rank["test_statistic"],
                "mann_whitney_raw_p_value": rank["raw_p_value"],
                "probability_of_superiority_churned": rank[
                    "probability_of_superiority"
                ],
                "cliffs_delta": rank["cliffs_delta"],
                "ties_present": rank["ties_present"],
            }
        )
    return pd.DataFrame(rows)


def _categorical_tests(
    data: pd.DataFrame, target: str, hypotheses: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for hypothesis_id, predictor in hypotheses.items():
        result = categorical_independence_test(data[predictor], data[target])
        rows.append({"hypothesis_id": hypothesis_id, "predictor": predictor, **result})
    return pd.DataFrame(rows)


def _adjusted_tests(
    numerical: pd.DataFrame,
    categorical: pd.DataFrame,
    alpha: float,
    fdr_threshold: float,
) -> pd.DataFrame:
    rows = []
    for row in numerical.itertuples(index=False):
        rows.extend(
            [
                {
                    "test_id": f"{row.hypothesis_id}-welch",
                    "hypothesis_id": row.hypothesis_id,
                    "predictor": row.predictor,
                    "test": "Welch independent t-test",
                    "designation": "primary",
                    "hypothesis_family": "primary_feature_churn",
                    "raw_p_value": row.welch_raw_p_value,
                    "effect_size": row.cohens_d,
                    "effect_size_measure": "Cohen's d",
                },
                {
                    "test_id": f"{row.hypothesis_id}-mann-whitney",
                    "hypothesis_id": row.hypothesis_id,
                    "predictor": row.predictor,
                    "test": "Mann-Whitney U",
                    "designation": "complementary",
                    "hypothesis_family": "complementary_numerical_ranks",
                    "raw_p_value": row.mann_whitney_raw_p_value,
                    "effect_size": row.cliffs_delta,
                    "effect_size_measure": "Cliff's delta",
                },
            ]
        )
    for row in categorical.itertuples(index=False):
        rows.append(
            {
                "test_id": f"{row.hypothesis_id}-chi-square",
                "hypothesis_id": row.hypothesis_id,
                "predictor": row.predictor,
                "test": "Pearson chi-square",
                "designation": "primary",
                "hypothesis_family": "primary_feature_churn",
                "raw_p_value": row.raw_p_value,
                "effect_size": row.bias_corrected_cramers_v,
                "effect_size_measure": "Bias-corrected Cramer's V",
            }
        )
    return adjust_hypothesis_families(pd.DataFrame(rows), alpha, fdr_threshold)


def run_statistical_analysis(
    config_path: str | Path = "params.yaml",
) -> dict[str, Any]:
    """Generate and atomically publish configured statistical-analysis artifacts."""
    config = load_config(config_path)
    data_config = config["data"]
    analysis_config = config["statistical_analysis"]
    input_path = Path(data_config["development_path"])
    output_path = Path(analysis_config["output_directory"])
    _validate_paths(input_path, output_path, data_config)
    if not input_path.is_file():
        raise FileNotFoundError(f"Development dataset not found: {input_path}")

    alpha = float(analysis_config["significance_level"])
    fdr_threshold = float(analysis_config["fdr_threshold"])
    confidence_level = float(analysis_config["confidence_level"])
    minimum_segment_size = int(analysis_config["minimum_segment_size"])
    vif_thresholds = list(analysis_config["vif_screening_thresholds"])
    registered = analysis_config["registered_predictors"]
    numerical_hypotheses = dict(registered["numerical"])
    categorical_hypotheses = dict(registered["categorical"])
    segment_features = list(registered["confidence_segments"])
    mixed_categorical = list(config["eda"]["categorical_features"])
    target, identifier = data_config["target_column"], data_config["id_column"]
    numerical_features = list(numerical_hypotheses.values())
    if identifier in numerical_features + mixed_categorical or target in (
        numerical_features + mixed_categorical
    ):
        raise StatisticalAnalysisError("Identifier and target cannot be predictors.")
    if not (0 < alpha < 1 and 0 < fdr_threshold < 1 and 0 < confidence_level < 1):
        raise StatisticalAnalysisError("Configured probability thresholds are invalid.")
    if minimum_segment_size < 1 or len(vif_thresholds) != 2:
        raise StatisticalAnalysisError("Segment and VIF thresholds are invalid.")

    data = pd.read_parquet(input_path)
    required = [
        identifier,
        target,
        *numerical_features,
        *categorical_hypotheses.values(),
        *segment_features,
        *mixed_categorical,
    ]
    _validate_columns(data, list(dict.fromkeys(required)))
    logger.info("Loaded development dataset with %d records", len(data))

    numerical = _numerical_tests(data, target, numerical_hypotheses, confidence_level)
    categorical = _categorical_tests(data, target, categorical_hypotheses)
    adjusted = _adjusted_tests(numerical, categorical, alpha, fdr_threshold)
    intervals = _confidence_intervals(
        data,
        target,
        numerical,
        segment_features,
        confidence_level,
        minimum_segment_size,
    )

    numerical_vif = calculate_vif(data[numerical_features].astype(float))
    numerical_vif.insert(0, "diagnostic_scope", "numerical_only")
    mixed_design, references = reference_coded_design(
        data, numerical_features, mixed_categorical
    )
    mixed_diagnostics = matrix_diagnostics(mixed_design)
    mixed_vif = calculate_vif(mixed_design)
    mixed_vif.insert(0, "diagnostic_scope", "reference_coded_mixed")
    multicollinearity = pd.concat([numerical_vif, mixed_vif], ignore_index=True)
    multicollinearity["diagnostic_interpretation"] = multicollinearity["vif"].map(
        lambda value: _interpret_vif(value, vif_thresholds)
    )

    family_summary: dict[str, dict[str, int]] = {}
    for family_name, family in adjusted.groupby("hypothesis_family", sort=False):
        family_summary[family_name] = {
            "test_count": len(family),
            "raw_rejections": int(family["raw_reject"].sum()),
            "bh_rejections": int(family["bh_reject"].sum()),
            "holm_rejections": int(family["holm_reject"].sum()),
            "underflow_bounds": int(family["p_value_underflow_bounded"].sum()),
        }
    overall_interval = intervals.loc[
        intervals["interval_id"].eq("overall_churn_rate")
    ].iloc[0]
    primary = adjusted.loc[adjusted["hypothesis_family"].eq("primary_feature_churn")]
    changed_bh = int((primary["raw_reject"] != primary["bh_reject"]).sum())
    mixed_condition = mixed_diagnostics["condition_number"]
    summary = {
        "schema_version": "1.0.0",
        "data_source": str(input_path),
        "data_source_sha256": _fingerprint(input_path),
        "dataset_dimensions": {"rows": len(data), "columns": data.shape[1]},
        "significance_level": alpha,
        "fdr_threshold": fdr_threshold,
        "confidence_level": confidence_level,
        "hypothesis_families": family_summary,
        "mixed_design": {
            **mixed_diagnostics,
            "condition_number": None
            if not np.isfinite(mixed_condition)
            else mixed_condition,
            "condition_number_status": "infinite_exact_rank_deficiency"
            if not np.isfinite(mixed_condition)
            else "finite",
            "reference_categories": references,
            "infinite_vif_columns": mixed_vif.loc[
                np.isinf(mixed_vif["vif"]), "predictor"
            ].tolist(),
        },
        "key_findings": {
            "overall_churn_rate": overall_interval["estimate"],
            "overall_churn_ci_lower": overall_interval["ci_lower"],
            "overall_churn_ci_upper": overall_interval["ci_upper"],
            "primary_bh_changes": changed_bh,
            "tenure_cohens_d": numerical.set_index("predictor").loc[
                "tenure", "cohens_d"
            ],
            "contract_cramers_v": categorical.set_index("predictor").loc[
                "Contract", "bias_corrected_cramers_v"
            ],
            "total_charges_vif": numerical_vif.set_index("predictor").loc[
                "TotalCharges", "vif"
            ],
            "mixed_design_rank": mixed_diagnostics["rank"],
            "mixed_design_columns_with_intercept": mixed_diagnostics[
                "diagnostic_columns_with_intercept"
            ],
            "mixed_design_infinite_vif_columns": int(np.isinf(mixed_vif["vif"]).sum()),
            "mixed_design_condition_number_display": "infinite (exact rank deficiency)"
            if not np.isfinite(mixed_condition)
            else f"{mixed_condition:.4f}",
        },
        "recommendations": [
            "Compare structural availability-plus-adoption encodings before "
            "coefficient-facing models.",
            "Evaluate tenure, MonthlyCharges, and TotalCharges redundancy with "
            "leakage-safe cross-validation.",
            "Use regularization for coefficient stability when the model family "
            "supports it.",
            "Do not remove predictors from marginal p-values or VIF alone.",
        ],
        "limitations": [
            "Development split only; EDA-informed inference is not independent "
            "confirmation.",
            "Associations and confidence intervals do not establish causal effects.",
            "VIF and dummy-column values depend on the temporary reference coding.",
            "No feature selection, persistent preprocessing, balancing, or model "
            "training is performed.",
        ],
    }
    tables = {
        "numerical_hypothesis_tests": numerical,
        "categorical_hypothesis_tests": categorical,
        "adjusted_hypothesis_tests": adjusted,
        "confidence_intervals": intervals,
        "multicollinearity": multicollinearity,
    }

    staging = output_path.with_name(output_path.name + ".staging")
    backup = output_path.with_name(output_path.name + ".previous")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        table_names = write_tables(tables, staging, index=False)
        write_summary(summary, staging / "summary.json")
        write_statistical_html(summary, table_names, staging / "report.html")
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
            "Statistical report generation failed; current report was preserved."
        )
        raise
    logger.info("Statistical report published to %s", output_path)
    return summary


def main() -> None:
    """CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Generate automated development-data statistical analysis."
    )
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    try:
        run_statistical_analysis(args.config)
    except Exception as exc:
        logger.error("Statistical analysis stage terminated: %s", type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
