"""Complementary fold-held-out explanations and diagnostic report generation."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/retention-matplotlib")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from xgboost import DMatrix

from retention.evaluation.diagnostics import OUTPUT, SOURCE, fit_fold, json_safe
from retention.evaluation.threshold import operating_points, select_constrained_recall
from retention.features.preprocessing import make_preprocessor, split_features_target
from retention.models.training import make_folds
from retention.utils.config import load_config


def paired_interval(differences):
    values = np.asarray(differences, dtype=float)
    mean = values.mean()
    half = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": mean, "low": mean - half, "high": mean + half}


def cross_model(X, y, p, folds, threshold):
    rows, errors = [], []
    base_error = (p >= threshold) != y
    base_ap = np.array([average_precision_score(y[v], p[v]) for _, v in folds])
    for model in ["logistic_regression", "random_forest", "xgboost", "ann"]:
        frame = pd.read_csv(
            Path("reports/generated/model_optimization") / f"{model}_oof.csv"
        )
        if len(frame) != len(y) or not np.array_equal(frame.observed_churn, y):
            raise ValueError(f"{model} OOF predictions misaligned")
        score = frame.oof_probability.to_numpy()
        point = select_constrained_recall(operating_points(y, score), 0.55)
        candidate_error = (
            (score >= point["threshold"]) != y
            if point
            else np.zeros(len(y), dtype=bool)
        )
        errors.append(
            {
                "model": model,
                "baseline_errors": int(base_error.sum()),
                "model_errors": int(candidate_error.sum()),
                "shared_errors": int((base_error & candidate_error).sum()),
                "baseline_only_errors": int((base_error & ~candidate_error).sum()),
            }
        )
        for alpha in [0.25]:
            blend = (1 - alpha) * p + alpha * score
            ap = np.array([average_precision_score(y[v], blend[v]) for _, v in folds])
            point = select_constrained_recall(operating_points(y, blend), 0.55)
            rows.append(
                {
                    "candidate": f"75% frozen + 25% {model}",
                    "mean_fold_ap": ap.mean(),
                    "pooled_ap": average_precision_score(y, blend),
                    "roc_auc": roc_auc_score(y, blend),
                    "precision": point["precision"] if point else np.nan,
                    "constrained_recall": point["recall"] if point else np.nan,
                    "f1": point["f1"] if point else np.nan,
                    **{
                        f"paired_ap_{k}": v
                        for k, v in paired_interval(ap - base_ap).items()
                    },
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "conditional_ensembles.csv", index=False)
    pd.DataFrame(errors).to_csv(OUTPUT / "cross_model_errors.csv", index=False)
    return rows, errors


def explanation_and_overlap(
    X, y, p, folds, params, weight, threshold, config_path="params.yaml"
):
    shap_rows, permutation_rows, neighbour_rows = [], [], []
    names = list(X.columns)
    for fold, (train, valid) in enumerate(folds, 1):
        pipe, raw, _ = fit_fold(
            X,
            pd.Series(y),
            train,
            valid,
            params,
            weight,
            42,
            config_path=config_path,
        )
        transformed = pipe.named_steps["preprocess"].transform(X.iloc[valid])
        model = pipe.named_steps["estimator"]
        booster = model.get_booster()
        sample = np.arange(min(120, len(valid)))
        matrix = DMatrix(transformed[sample], feature_names=None)
        contributions = booster.predict(matrix, pred_contribs=True)
        margins = booster.predict(matrix, output_margin=True)
        if not np.allclose(contributions.sum(axis=1), margins, atol=1e-4):
            raise ValueError("XGBoost margin SHAP additivity failed")
        encoded = pipe.named_steps["preprocess"].get_feature_names_out()
        for position, row in enumerate(contributions):
            original = valid[sample[position]]
            group = (
                ("TP" if y[original] else "FP")
                if p[original] >= threshold
                else ("FN" if y[original] else "TN")
            )
            for feature in names:
                mask = np.array(
                    [
                        name == f"numeric__{feature}"
                        or name == f"binary__{feature}"
                        or name.startswith(f"categorical__{feature}_")
                        for name in encoded
                    ]
                )
                shap_rows.append(
                    {
                        "fold": fold,
                        "group": group,
                        "feature": feature,
                        "margin_contribution": float(row[:-1][mask].sum()),
                        "absolute_contribution": float(np.abs(row[:-1][mask]).sum()),
                    }
                )
        baseline_ap = average_precision_score(y[valid], raw)
        rng = np.random.default_rng(100 + fold)
        for feature in names:
            drops = []
            for _ in range(3):
                shuffled = X.iloc[valid].copy()
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                score = pipe.predict_proba(shuffled)[:, 1]
                drops.append(baseline_ap - average_precision_score(y[valid], score))
            permutation_rows.append(
                {"fold": fold, "feature": feature, "ap_drop": np.mean(drops)}
            )
        # Standardize numerical columns on the training fold before distance.
        prep = make_preprocessor("baseline", "linear", config_path=config_path)
        train_matrix = prep.fit_transform(X.iloc[train])
        valid_matrix = prep.transform(X.iloc[valid])
        nn = NearestNeighbors(n_neighbors=1, metric="manhattan").fit(train_matrix)
        distance, nearest = nn.kneighbors(valid_matrix)
        neighbour_rows.extend(
            {
                "fold": fold,
                "distance": float(d),
                "class_disagrees": bool(y[v] != y[train[n]]),
            }
            for v, d, n in zip(valid, distance[:, 0], nearest[:, 0], strict=True)
        )
    shap = pd.DataFrame(shap_rows)
    shap.groupby(["group", "feature"])[
        ["margin_contribution", "absolute_contribution"]
    ].mean().reset_index().to_csv(OUTPUT / "shap_error_groups.csv", index=False)
    pd.DataFrame(permutation_rows).to_csv(
        OUTPUT / "permutation_importance.csv", index=False
    )
    pd.DataFrame(neighbour_rows).to_csv(
        OUTPUT / "nearest_neighbour_overlap.csv", index=False
    )
    return {
        "shap_rows": len(shap) // len(names),
        "shap_additivity": True,
        "nearest_class_disagreement": float(
            pd.DataFrame(neighbour_rows).class_disagrees.mean()
        ),
        "nearest_distance_median": float(
            pd.DataFrame(neighbour_rows).distance.median()
        ),
    }


def finish(config_path="params.yaml"):
    config = load_config(config_path)
    data = pd.read_parquet(config["data"]["development_path"])
    X, ys = split_features_target(data, config_path)
    y = ys.to_numpy()
    folds = make_folds(ys, 5, config["project"]["random_state"])
    selection = json.loads((SOURCE / "revised_selection.json").read_text())
    p = pd.read_csv(SOURCE / "calibration_oof.csv").sigmoid_probability.to_numpy()
    findings = json.loads((OUTPUT / "findings.json").read_text())
    internet_dependent = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    offline = X.InternetService.eq("No")
    structural_violations = int(
        (~X.loc[offline, internet_dependent].eq("No internet service")).to_numpy().sum()
    )
    findings["data_quality"] = {
        "missing_predictor_cells": int(X.isna().to_numpy().sum()),
        "zero_tenure_rows": int(X.tenure.eq(0).sum()),
        "offline_service_dependency_violations": structural_violations,
    }
    profiles = X.copy()
    profiles["Churn"] = y
    repeats = profiles.groupby(list(X.columns), dropna=False).Churn.agg(
        ["size", "nunique"]
    )
    contradictory = repeats[repeats["nunique"] > 1].reset_index()
    contradictory[
        [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "InternetService",
            "PaymentMethod",
            "size",
        ]
    ].head(10).to_csv(OUTPUT / "contradictory_profiles_anonymized.csv", index=False)
    predicted = p >= selection["threshold"]
    groups = np.select(
        [(y == 1) & predicted, (y == 0) & ~predicted, (y == 0) & predicted],
        ["TP", "TN", "FP"],
        default="FN",
    )
    probability_groups = (
        pd.DataFrame(
            {
                "group": groups,
                "raw": pd.read_csv(SOURCE / "calibration_oof.csv").raw_probability,
                "calibrated": p,
            }
        )
        .groupby("group")[["raw", "calibrated"]]
        .agg(["count", "mean", "median", "min", "max"])
    )
    probability_groups.columns = [
        f"{score}_{statistic}" for score, statistic in probability_groups.columns
    ]
    probability_groups.reset_index().to_csv(
        OUTPUT / "probability_error_groups.csv", index=False
    )
    effects = []
    for feature in ["tenure", "MonthlyCharges", "TotalCharges"]:
        values = X[feature].to_numpy(dtype=float)
        for error, reference in [("FN", "TP"), ("FP", "TN")]:
            left, right = values[groups == error], values[groups == reference]
            left_mean = float(left.mean()) if len(left) else np.nan
            right_mean = float(right.mean()) if len(right) else np.nan
            pooled = (
                np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2)
                if min(len(left), len(right)) >= 2
                else np.nan
            )
            effects.append(
                {
                    "feature": feature,
                    "error_group": error,
                    "reference_group": reference,
                    "error_mean": left_mean,
                    "reference_mean": right_mean,
                    "standardized_mean_difference": (
                        (left_mean - right_mean) / pooled
                        if np.isfinite(pooled) and pooled > 0
                        else np.nan
                    ),
                }
            )
    pd.DataFrame(effects).to_csv(OUTPUT / "error_effect_sizes.csv", index=False)
    blends, errors = cross_model(X, y, p, folds, selection["threshold"])
    extra = explanation_and_overlap(
        X,
        y,
        p,
        folds,
        selection["parameters"],
        selection["scale_pos_weight"],
        selection["threshold"],
        config_path,
    )
    findings.update(
        {
            "cross_model_errors": errors,
            "conditional_ensembles": blends,
            "overlap_and_explanation": extra,
        }
    )
    ablation = pd.read_csv(OUTPUT / "ablations.csv")
    baseline_ap = np.array([average_precision_score(y[v], p[v]) for _, v in folds])
    findings["ablation_paired_ap"] = {
        name: paired_interval(frame.sort_values("fold").ap.to_numpy() - baseline_ap)
        for name, frame in ablation.groupby("group")
    }
    findings["recommendation"] = "retain_existing"
    findings["selection_optimism"] = (
        "Repeated development-fold inspection and threshold selection "
        "can bias apparent gains upward."
    )
    (OUTPUT / "findings.json").write_text(
        json.dumps(json_safe(findings), indent=2, allow_nan=False)
    )
    pd.DataFrame(
        [
            {
                "observed_limitation": "Depth raises training AP, lowers validation AP",
                "intervention": "Controlled depth 2-5",
                "decision": "Reject",
            },
            {
                "observed_limitation": "More than 535 trees give no AP gain",
                "intervention": "750 and 1000 trees",
                "decision": "Reject",
            },
            {
                "observed_limitation": "Charges correlate with tenure and services",
                "intervention": "Remove charge group",
                "decision": "Reject: gain below margin",
            },
            {
                "observed_limitation": "Other models make some different errors",
                "intervention": "25% conditional blends",
                "decision": "Reject: gain below margin",
            },
            {
                "observed_limitation": "Learning curve nearly plateaus",
                "intervention": "Better customer behavior data",
                "decision": "Recommend future research",
            },
        ]
    ).to_csv(OUTPUT / "diagnostic_findings_table.csv", index=False)
    seg = pd.read_csv(OUTPUT / "segments.csv")
    important = (
        seg[
            (seg.n >= 100)
            & (
                seg.segment.isin(
                    [
                        "Contract",
                        "tenure_band",
                        "InternetService",
                        "PaymentMethod",
                        "TechSupport",
                        "OnlineSecurity",
                    ]
                )
            )
        ]
        .sort_values("fn", ascending=False)
        .head(12)
    )
    curves = pd.read_csv(OUTPUT / "learning_capacity_curves.csv")
    summary = (
        curves.groupby(["curve", "setting", "split"])[
            ["ap", "roc_auc", "log_loss", "constrained_recall"]
        ]
        .mean()
        .reset_index()
    )
    precision, recall, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    axes[0].plot(recall, precision)
    axes[0].axhline(0.55, linestyle="--", color="black")
    axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    for axis, (curve_name, label) in zip(
        axes[1:],
        [("learning", "Learning"), ("depth", "Depth"), ("trees", "Trees")],
        strict=True,
    ):
        subset = summary[
            (summary.curve == curve_name) & (summary.split == "validation")
        ]
        axis.plot(subset.setting, subset.ap, marker="o")
        axis.set(xlabel=label, ylabel="Validation AP")
    fig.tight_layout()
    fig.savefig(OUTPUT / "ranking_capacity.png", dpi=140)
    plt.close(fig)
    comparison = pd.read_csv(OUTPUT / "candidate_comparison.csv")
    html = [
        (
            "<html><head><meta charset='utf-8'><title>Phase 6A++ diagnostics"
            "</title><style>body{font:16px sans-serif;max-width:1100px;"
            "margin:40px auto}"
            "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;"
            "padding:5px}img{max-width:100%}</style></head><body>"
        ),
        "<h1>Phase 6A++ model diagnostics</h1>",
        (
            "<p>Development-only OOF investigation. Frozen sigmoid XGBoost retained."
            " No held-out evaluation or final refit.</p>"
        ),
        "<h2>Frozen operating point</h2>",
        pd.DataFrame([findings["baseline"]]).round(4).to_html(index=False),
        "<p>Average precision summarizes the full ranking. It does not isolate "
        "the high-recall region where precision must remain at least 0.55.</p>",
        "<h2>Largest error sources</h2>",
        important[
            ["segment", "value", "n", "fn", "fp", "fnr", "fnr_low", "fnr_high", "fpr"]
        ]
        .round(3)
        .to_html(index=False),
        (
            "<h2>Probability overlap and reliability</h2>"
            "<img src='probability_overlap_reliability.png'>"
        ),
        "<h2>Learning and capacity curves</h2>",
        "<img src='ranking_capacity.png'>",
        summary.round(4).to_html(index=False),
        "<h2>Capacity candidates</h2>",
        comparison.round(4).to_html(index=False),
        "<h2>Ablation paired AP differences</h2>",
        pd.DataFrame(findings["ablation_paired_ap"]).T.round(4).to_html(),
        "<h2>Conditional ensembles</h2>",
        pd.DataFrame(blends).round(4).to_html(index=False),
        "<h2>Data quality and overlap</h2>",
        pd.DataFrame([findings["duplicates"]]).to_html(index=False),
        "<p>Identical recorded profiles can have different outcomes. The nearest "
        "neighbour disagreement is descriptive, not an irreducible error estimate.</p>",
        "<h2>Recommendation</h2>",
        "<p>Retain the frozen sigmoid-calibrated depth-one XGBoost. More depth "
        "worsens validation AP, extra trees plateau, and tested blends do not "
        "clear the improvement margin. Better behavioural or longitudinal data "
        "is a more promising next research direction than added model complexity.</p>",
        (
            "<p>SHAP contributions explain raw XGBoost margins, not calibrated "
            "probabilities. Correlated predictors affect importance estimates. "
            "Associations are not causal. Development-fold selection is optimistic.</p>"
        ),
        ("<p>Data tables are saved alongside this report.</p></body></html>"),
    ]
    (OUTPUT / "report.html").write_text("\n".join(html))
    (OUTPUT / "selection_report.json").write_text(
        json.dumps(
            {
                "decision": "retain_existing",
                "reason": (
                    "Greater depth reduced mean fold AP; no tested intervention "
                    "cleared the engineering improvement margin consistently."
                ),
                "frozen_source": str(SOURCE / "revised_selection.json"),
                "selection_optimism": findings["selection_optimism"],
            },
            indent=2,
        )
    )
    mlflow.set_tracking_uri(config["model_training"]["mlflow_tracking_uri"])
    with mlflow.start_run(run_id=findings["mlflow_run_id"]):
        mlflow.log_artifacts(str(OUTPUT))
    return findings


if __name__ == "__main__":
    finish()
