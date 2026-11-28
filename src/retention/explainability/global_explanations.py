"""Reproducible global raw-margin SHAP summaries from development predictors only."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from retention.explainability.service import frozen_explanation_service  # noqa: E402

DEVELOPMENT = Path("data/processed/development.parquet")
OUTPUT = Path("reports/generated/explainability")
SEED = 42
MAX_SAMPLE = 1000
CHARTS = (
    "global_importance.png",
    "beeswarm.png",
    "dependence_tenure.png",
    "dependence_contract.png",
    "dependence_internet_service.png",
)


def _save_bar(importance: list[dict], path: Path) -> None:
    ordered = list(reversed(importance))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        [item["feature"] for item in ordered],
        [item["mean_absolute_raw_margin_contribution"] for item in ordered],
        color="#28658a",
    )
    ax.set_xlabel("Mean |SHAP contribution| to raw XGBoost margin")
    ax.set_title("Global original-feature importance (development sample)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_beeswarm(matrix: np.ndarray, sample: pd.DataFrame, path: Path) -> None:
    explanation = shap.Explanation(
        values=matrix,
        data=sample.to_numpy(),
        feature_names=list(sample.columns),
    )
    shap.plots.beeswarm(explanation, max_display=19, show=False, plot_size=(10, 8))
    plt.xlabel("SHAP contribution to raw XGBoost margin")
    plt.title("Original-feature effects (development sample)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_dependence(
    feature: str, values: pd.Series, effects: np.ndarray, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if feature == "tenure":
        ax.scatter(values.to_numpy(dtype=float), effects, s=11, alpha=0.35)
        ax.set_xlabel("Tenure (months)")
    else:
        categories = sorted(values.astype(str).unique())
        rng = np.random.default_rng(SEED)
        for position, category in enumerate(categories):
            mask = values.astype(str).eq(category).to_numpy()
            x = position + rng.uniform(-0.18, 0.18, mask.sum())
            ax.scatter(x, effects[mask], s=11, alpha=0.35, label=category)
        ax.set_xticks(range(len(categories)), categories, rotation=15)
        ax.set_xlabel(feature)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Original-feature SHAP contribution to raw margin")
    ax.set_title(f"SHAP dependence: {feature}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_global_explanations(
    output: Path = OUTPUT, development: Path = DEVELOPMENT
) -> dict:
    service = frozen_explanation_service()
    actual_development_sha = hashlib.sha256(development.read_bytes()).hexdigest()
    if actual_development_sha != service.predictor.metadata["development_sha256"]:
        raise ValueError("Development dataset fingerprint mismatch")
    input_frame = pd.read_parquet(development, columns=list(service.predictor.columns))
    sample = input_frame.sample(
        n=min(MAX_SAMPLE, len(input_frame)), random_state=SEED
    ).reset_index(drop=True)
    sample_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "development_sha256": actual_development_sha,
                "seed": SEED,
                "rows": len(sample),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    fingerprints = {
        "model_sha256": service.model_sha256,
        "configuration_sha256": service.predictor.metadata["configuration_sha256"],
        "explainer_sha256": service.fingerprint,
        "development_sha256": actual_development_sha,
        "sample_sha256": sample_fingerprint,
    }
    summary_path = output / "summary.json"
    if summary_path.exists() and all((output / chart).exists() for chart in CHARTS):
        previous = json.loads(summary_path.read_text())
        if previous.get("fingerprints") == fingerprints:
            return previous
    records = sample.to_dict(orient="records")
    explanations = []
    for start in range(0, len(records), 100):
        explanations.extend(service.explain_many(records[start : start + 100]))
    names = service.predictor.columns
    matrix = np.asarray(
        [
            [
                item.raw_margin_contribution
                for item in result.original_feature_contributions
            ]
            for result in explanations
        ],
        dtype=float,
    )
    if matrix.shape != (len(sample), len(names)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid global SHAP matrix")
    importance = [
        {
            "feature": name,
            "mean_absolute_raw_margin_contribution": float(np.abs(matrix[:, i]).mean()),
            "mean_signed_raw_margin_contribution": float(matrix[:, i].mean()),
            "p05_raw_margin_contribution": float(np.quantile(matrix[:, i], 0.05)),
            "p50_raw_margin_contribution": float(np.quantile(matrix[:, i], 0.50)),
            "p95_raw_margin_contribution": float(np.quantile(matrix[:, i], 0.95)),
        }
        for i, name in enumerate(names)
    ]
    importance.sort(
        key=lambda item: (
            -item["mean_absolute_raw_margin_contribution"],
            item["feature"],
        )
    )
    summary = {
        "fingerprints": fingerprints,
        "method": "tree_shap",
        "feature_perturbation": "tree_path_dependent",
        "explained_output_space": "raw_xgboost_margin",
        "sample_rows": len(sample),
        "random_seed": SEED,
        "importance": importance,
        "charts": list(CHARTS),
    }
    output.mkdir(parents=True, exist_ok=True)
    _save_bar(importance, output / CHARTS[0])
    _save_beeswarm(matrix, sample.loc[:, names], output / CHARTS[1])
    for feature, chart in (
        ("tenure", CHARTS[2]),
        ("Contract", CHARTS[3]),
        ("InternetService", CHARTS[4]),
    ):
        _save_dependence(
            feature, sample[feature], matrix[:, names.index(feature)], output / chart
        )
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary = generate_global_explanations(output=args.output)
    print(
        json.dumps(
            {
                "sample_rows": summary["sample_rows"],
                "top_features": [item["feature"] for item in summary["importance"][:5]],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
