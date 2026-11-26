"""Concise, deterministic PNG figures for recurring EDA."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/retention-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = ["#4C78A8", "#E45756", "#59A14F", "#F28E2B", "#B279A2"]


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        path, dpi=dpi, bbox_inches="tight", metadata={"Software": "retention-eda"}
    )
    plt.close(fig)


def generate_figures(
    df: pd.DataFrame,
    output: Path,
    numerical: list[str],
    target: str,
    correlations: pd.DataFrame,
    categorical_assoc: pd.DataFrame,
    mutual_information: pd.DataFrame,
    lifecycle: pd.DataFrame,
    risk_value: pd.DataFrame,
    contract: pd.DataFrame,
    outlier_methods: pd.DataFrame,
    pareto: pd.DataFrame,
    dpi: int,
    width: float,
    height: float,
) -> list[str]:
    """Generate the configured compact operational figure set."""
    plt.rcParams.update({"figure.dpi": dpi, "axes.grid": True, "axes.axisbelow": True})
    generated: list[str] = []

    def save(name: str, fig: plt.Figure) -> None:
        _save(fig, output / name, dpi)
        generated.append(name)

    counts = df[target].value_counts().reindex(["No", "Yes"])
    fig, ax = plt.subplots(figsize=(width, height))
    counts.plot.bar(ax=ax, color=COLORS[:2])
    ax.set(title="Customer outcome distribution", xlabel="", ylabel="Customers")
    save("target_distribution.png", fig)

    fig, axes = plt.subplots(1, len(numerical), figsize=(width * 2, height))
    for ax, feature in zip(np.atleast_1d(axes), numerical, strict=True):
        ax.hist(df[feature], bins=30, color=COLORS[0], edgecolor="white")
        ax.set(title=feature, ylabel="Customers")
    save("numerical_distributions.png", fig)

    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(
        range(len(correlations)), correlations.columns, rotation=30, ha="right"
    )
    ax.set_yticks(range(len(correlations)), correlations.index)
    for i in range(len(correlations)):
        for j in range(len(correlations)):
            ax.text(j, i, f"{correlations.iloc[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    ax.set_title("Pearson correlation")
    save("numerical_correlation.png", fig)

    for name, table, column, title in [
        (
            "categorical_target_association.png",
            categorical_assoc,
            "bias_corrected_v",
            "Categorical association with churn",
        ),
        (
            "mutual_information.png",
            mutual_information,
            "mutual_information_nats",
            "Exploratory mutual information",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(width, height * 1.5))
        table[column].sort_values().plot.barh(ax=ax, color=COLORS[2])
        ax.set(title=title, xlabel=column, ylabel="")
        save(name, fig)

    category = contract.sort_values("churn_rate")
    fig, ax = plt.subplots(figsize=(width, height))
    category["churn_rate"].plot.bar(ax=ax, color=COLORS[1])
    ax.set(title="Churn rate by contract", xlabel="", ylabel="Observed churn rate")
    save("category_churn_rates.png", fig)

    fig, ax = plt.subplots(figsize=(width, height))
    lifecycle["churn_rate"].plot.bar(ax=ax, color=COLORS[1])
    ax.set(title="Lifecycle churn rate", xlabel="", ylabel="Observed churn rate")
    save("lifecycle_segmentation.png", fig)

    pivot = risk_value.pivot(
        index="Contract", columns="tenure_segment", values="churn_rate"
    ).astype(float)
    fig, ax = plt.subplots(figsize=(width * 1.4, height))
    image = ax.imshow(pivot, vmin=0, vmax=max(0.01, np.nanmax(pivot)), cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1]), [str(x).split(":")[0] for x in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    fig.colorbar(image, ax=ax)
    ax.set(
        title="Contract × lifecycle churn rate",
        xlabel="Tenure segment",
        ylabel="Contract",
    )
    save("contract_lifecycle_heatmap.png", fig)

    fig, ax = plt.subplots(figsize=(width, height))
    contract["total_monthly_charge_exposure"].plot.bar(ax=ax, color=COLORS[0])
    ax.set(
        title="Monthly-charge exposure by contract",
        xlabel="",
        ylabel="Charge-field total",
    )
    save("monthly_charge_exposure_by_contract.png", fig)

    fig, ax = plt.subplots(figsize=(width, height))
    outlier_methods.sum().plot.bar(ax=ax, color=COLORS[3])
    ax.set(title="Diagnostic outlier screening", xlabel="", ylabel="Rows flagged")
    save("outlier_screening.png", fig)

    fig, ax = plt.subplots(figsize=(width * 1.4, height))
    x = np.arange(len(pareto))
    ax.bar(x, pareto["churned_count"], color=COLORS[0])
    ax.set_xticks(x, pareto["segment_label"], rotation=55, ha="right")
    ax.set_ylabel("Churned customers")
    second = ax.twinx()
    second.plot(x, pareto["cumulative_contribution"], color=COLORS[1], marker="o")
    second.axhline(0.8, color="black", linestyle="--")
    second.set_ylabel("Cumulative contribution")
    ax.set_title("Pareto concentration of churn volume")
    save("pareto_concentration.png", fig)
    return generated
