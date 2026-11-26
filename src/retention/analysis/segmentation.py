"""Descriptive customer segmentation and business exposure summaries."""

from typing import Any

import numpy as np
import pandas as pd

LIFECYCLE_LABELS = [
    "0–6: Onboarding",
    "7–12: First year",
    "13–24: Developing",
    "25–48: Established",
    "49–72: Mature",
]


def prepare_segments(
    df: pd.DataFrame, target: str, tenure_boundaries: list[float]
) -> pd.DataFrame:
    """Create notebook-aligned lifecycle, value, service, and outcome fields."""
    data = df.copy()
    data["_churned"] = data[target].eq("Yes")
    data["_churn_monthly_exposure"] = data["MonthlyCharges"].where(
        data["_churned"], 0.0
    )
    data["tenure_segment"] = pd.cut(
        data["tenure"],
        bins=tenure_boundaries,
        labels=LIFECYCLE_LABELS,
        include_lowest=True,
        ordered=True,
    )
    quartiles = data["MonthlyCharges"].quantile([0.25, 0.50, 0.75]).tolist()
    data["value_segment"] = pd.cut(
        data["MonthlyCharges"],
        bins=[-np.inf, *quartiles, np.inf],
        labels=["Q1 Lower", "Q2 Lower-middle", "Q3 Upper-middle", "Q4 Higher"],
        include_lowest=True,
        ordered=True,
    )
    addons = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    data["service_count"] = (
        data["PhoneService"].eq("Yes").astype(int)
        + data["MultipleLines"].eq("Yes").astype(int)
        + data["InternetService"].ne("No").astype(int)
        + sum(data[x].eq("Yes").astype(int) for x in addons)
    )
    data["adoption_band"] = pd.cut(
        data["service_count"],
        bins=[-0.1, 2, 5, 9],
        labels=["Low (0–2)", "Moderate (3–5)", "High (6–9)"],
        ordered=True,
    )
    return data


def business_kpis(data: pd.DataFrame) -> dict[str, Any]:
    """Return observed population, churn, and monthly-charge exposure KPIs."""
    churned = data["_churned"]
    total_exposure = float(data["MonthlyCharges"].sum())
    churn_exposure = float(data.loc[churned, "MonthlyCharges"].sum())
    return {
        "customers": int(len(data)),
        "churned_customers": int(churned.sum()),
        "retained_customers": int((~churned).sum()),
        "churn_rate": float(churned.mean()),
        "retention_rate": float((~churned).mean()),
        "total_monthly_charge_exposure": total_exposure,
        "churn_associated_monthly_charge_exposure": churn_exposure,
        "churn_associated_exposure_share": churn_exposure / total_exposure
        if total_exposure
        else None,
        "average_monthly_charges": float(data["MonthlyCharges"].mean()),
        "median_tenure": float(data["tenure"].median()),
        "total_accumulated_charges": float(data["TotalCharges"].sum()),
    }


def segment_summary(data: pd.DataFrame, groups: str | list[str]) -> pd.DataFrame:
    """Describe mutually exclusive segments and their churn-rate lift."""
    overall_rate = float(data["_churned"].mean())
    total_churned = int(data["_churned"].sum())
    result = data.groupby(groups, observed=True, dropna=False).agg(
        customer_count=("_churned", "size"),
        churned_count=("_churned", "sum"),
        mean_monthly_charges=("MonthlyCharges", "mean"),
        median_monthly_charges=("MonthlyCharges", "median"),
        total_monthly_charge_exposure=("MonthlyCharges", "sum"),
        churn_monthly_charge_exposure=("_churn_monthly_exposure", "sum"),
    )
    result["retained_count"] = result["customer_count"] - result["churned_count"]
    result["churn_rate"] = result["churned_count"] / result["customer_count"]
    result["share_of_all_churners"] = (
        result["churned_count"] / total_churned if total_churned else np.nan
    )
    result["churn_rate_lift"] = (
        result["churn_rate"] / overall_rate if overall_rate else np.nan
    )
    return result


def verify_reconciliation(data: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Raise if segment counts, churn volumes, or exposures fail to reconcile."""
    checks = (
        int(summary["customer_count"].sum()) == len(data),
        int(summary["churned_count"].sum()) == int(data["_churned"].sum()),
        np.isclose(
            summary["total_monthly_charge_exposure"].sum(), data["MonthlyCharges"].sum()
        ),
        np.isclose(
            summary["churn_monthly_charge_exposure"].sum(),
            data["_churn_monthly_exposure"].sum(),
        ),
    )
    if not all(checks):
        raise ValueError(
            "Segment totals do not reconcile with the development population."
        )


def risk_value_segments(data: pd.DataFrame, minimum_sample_size: int) -> pd.DataFrame:
    """Classify contract-lifecycle groups using observed risk and charge proxies."""
    table = segment_summary(data, ["Contract", "tenure_segment"]).reset_index()
    table["sample_sufficient"] = table["customer_count"].ge(minimum_sample_size)
    table["observed_risk_band"] = np.where(
        table["churn_rate"].ge(data["_churned"].mean()),
        "High observed risk",
        "Low observed risk",
    )
    table["value_band"] = np.where(
        table["mean_monthly_charges"].ge(data["MonthlyCharges"].mean()),
        "High monthly-charge value",
        "Low monthly-charge value",
    )
    table["risk_value_quadrant"] = np.where(
        table["sample_sufficient"],
        table["observed_risk_band"] + " / " + table["value_band"],
        f"Insufficient sample (<{minimum_sample_size})",
    )
    return table


def pareto_concentration(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank mutually exclusive contract-lifecycle segments by contribution."""
    ranked = table.sort_values(
        [metric, "Contract", "tenure_segment"], ascending=[False, True, True]
    ).copy()
    ranked["segment_label"] = (
        ranked["Contract"].astype(str) + " | " + ranked["tenure_segment"].astype(str)
    )
    total = ranked[metric].sum()
    ranked["cumulative_contribution"] = (
        ranked[metric].cumsum() / total if total else np.nan
    )
    ranked["cumulative_segment_share"] = (
        (np.arange(len(ranked)) + 1) / len(ranked) if len(ranked) else np.nan
    )
    return ranked
