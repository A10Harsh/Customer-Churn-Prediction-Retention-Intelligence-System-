"""Development-only, row-aligned diagnostics for the frozen Phase 6A+ model."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/retention-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from retention.evaluation.metrics import binary_metrics
from retention.evaluation.threshold import operating_points, select_constrained_recall
from retention.features.preprocessing import make_preprocessor, split_features_target
from retention.models.factory import make_model
from retention.models.optimization import _fold_hash
from retention.models.training import _checksum, make_folds
from retention.models.xgboost_enhancement import fit_xgb_pipeline
from retention.utils.config import load_config

SOURCE = Path("reports/generated/xgboost_enhancement")
OUTPUT = Path("reports/generated/model_diagnostics")
GROUPS = {
    "contract_lifecycle": ["Contract", "tenure"],
    "charges_spending": ["MonthlyCharges", "TotalCharges"],
    "internet_services": [
        "InternetService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ],
    "payment": ["PaymentMethod", "PaperlessBilling"],
    "demographics": ["gender", "SeniorCitizen", "Partner", "Dependents"],
}


def wilson_interval(successes, trials):
    """Return a 95% Wilson interval without an optional dependency."""
    if trials == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = (
        z * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials**2)) / denominator
    )
    return center - radius, center + radius


def json_safe(value):
    """Convert non-finite diagnostic metrics to JSON null."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def verify_alignment(data, selection, calibration, raw_oof, folds, path):
    """Verify saved predictions and folds match development rows."""
    if _checksum(path) != selection["development_sha256"]:
        raise ValueError("Development checksum differs from frozen selection")
    if _fold_hash(folds) != selection["fold_sha256"]:
        raise ValueError("Fold assignments differ from frozen selection")
    if len(data) != len(calibration) or len(data) != len(raw_oof):
        raise ValueError("OOF row count mismatch")
    labels = data.Churn.map({"No": 0, "Yes": 1}).to_numpy()
    if not np.array_equal(
        labels, calibration.observed_churn.to_numpy()
    ) or not np.array_equal(labels, raw_oof.observed_churn.to_numpy()):
        raise ValueError("OOF labels are not row aligned")
    expected = np.full(len(data), -1)
    for fold, (_, valid) in enumerate(folds, 1):
        if (expected[valid] != -1).any():
            raise ValueError("Overlapping validation folds")
        expected[valid] = fold
    if not np.array_equal(expected, raw_oof.fold.to_numpy()):
        raise ValueError("OOF fold IDs are not row aligned")
    if not np.allclose(calibration.raw_probability, raw_oof.oof_probability, atol=1e-7):
        raise ValueError("Raw calibration scores do not match selected OOF scores")
    return labels, expected


def safe_metrics(y, p, threshold):
    """Return segment metrics with undefined values explicit."""
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    fn, tn = int(((y == 1) & ~pred).sum()), int(((y == 0) & ~pred).sum())
    pos, neg = tp + fn, tn + fp
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / pos if pos else np.nan
    return {
        "n": len(y),
        "churn_prevalence": pos / len(y) if len(y) else np.nan,
        "ap": average_precision_score(y, p) if pos and neg else np.nan,
        "roc_auc": roc_auc_score(y, p) if pos and neg else np.nan,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall
        else np.nan,
        "fnr": fn / pos if pos else np.nan,
        "fpr": fp / neg if neg else np.nan,
        "brier": brier_score_loss(y, p),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fnr_low": wilson_interval(fn, pos)[0],
        "fnr_high": wilson_interval(fn, pos)[1],
        "fpr_low": wilson_interval(fp, neg)[0],
        "fpr_high": wilson_interval(fp, neg)[1],
        "sparse": len(y) < 30 or min(pos, neg) < 10,
    }


def segments(X, y, p, threshold):
    predicted = p >= threshold
    total_fn = int(((y == 1) & ~predicted).sum())
    total_fp = int(((y == 0) & predicted).sum())
    frame = X.copy()
    frame["tenure_band"] = pd.cut(
        frame.tenure,
        [-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    )
    frame["monthly_band"] = pd.qcut(frame.MonthlyCharges, 4, duplicates="drop")
    frame["total_band"] = pd.qcut(frame.TotalCharges, 4, duplicates="drop")
    frame["support_security"] = (
        frame.TechSupport.astype(str) + " / " + frame.OnlineSecurity.astype(str)
    )
    keys = [
        "tenure_band",
        "Contract",
        "monthly_band",
        "total_band",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
        "SeniorCitizen",
        "gender",
        "Partner",
        "Dependents",
    ]
    pairs = [
        ("Contract", "tenure_band"),
        ("InternetService", "monthly_band"),
        ("PaymentMethod", "tenure_band"),
        ("TechSupport", "OnlineSecurity"),
    ]
    rows = []
    for key in keys + pairs:
        name = " × ".join(key) if isinstance(key, tuple) else key
        values = (
            frame[list(key)].astype(str).agg(" × ".join, axis=1)
            if isinstance(key, tuple)
            else frame[key].astype(str)
        )
        for value, idx in values.groupby(values).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            row = safe_metrics(y[idx], p[idx], threshold)
            row.update(
                {
                    "segment": name,
                    "value": str(value),
                    "fn_share": row["fn"] / total_fn if total_fn else np.nan,
                    "fp_share": row["fp"] / total_fp if total_fp else np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def fit_fold(
    X, y, train, valid, params, weight, seed, columns=None, config_path="params.yaml"
):
    columns = list(X.columns) if columns is None else columns
    if columns == list(X.columns):
        pipe, _, _ = fit_xgb_pipeline(
            X.iloc[train],
            y.iloc[train],
            {},
            "baseline",
            params,
            weight,
            None,
            False,
            seed,
            config_path,
        )
    else:
        pipe = Pipeline(
            [
                (
                    "preprocess",
                    make_preprocessor(
                        "baseline",
                        "tree",
                        selected_features=columns,
                        config_path=config_path,
                    ),
                ),
                (
                    "estimator",
                    make_model(
                        "xgboost",
                        "weighted",
                        seed,
                        params,
                        weight,
                    ),
                ),
            ]
        )
        pipe.fit(X.iloc[train], y.iloc[train])
    return (
        pipe,
        pipe.predict_proba(X.iloc[valid][columns])[:, 1],
        pipe.predict_proba(X.iloc[train][columns])[:, 1],
    )


def curve_experiments(X, y, folds, params, weight, config_path="params.yaml"):
    rows, ablations, predictions = [], [], {}
    for number, (train, valid) in enumerate(folds, 1):
        for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
            subset = (
                train
                if frac == 1
                else train_test_split(
                    train,
                    train_size=frac,
                    random_state=42 + number,
                    stratify=y.iloc[train],
                )[0]
            )
            _, pv, pt = fit_fold(
                X,
                y,
                subset,
                valid,
                params,
                weight,
                42,
                config_path=config_path,
            )
            for split, labels, scores in [
                ("train", y.iloc[subset], pt),
                ("validation", y.iloc[valid], pv),
            ]:
                rows.append(
                    {
                        "curve": "learning",
                        "setting": frac,
                        "fold": number,
                        "split": split,
                        "ap": average_precision_score(labels, scores),
                        "roc_auc": roc_auc_score(labels, scores),
                        "log_loss": log_loss(labels, scores),
                    }
                )
        for curve, settings in [
            ("depth", [1, 2, 3, 4, 5]),
            ("trees", [100, 250, 535, 750, 1000]),
        ]:
            for setting in settings:
                candidate = dict(params)
                candidate["max_depth" if curve == "depth" else "n_estimators"] = setting
                _, pv, pt = fit_fold(
                    X,
                    y,
                    train,
                    valid,
                    candidate,
                    weight,
                    42,
                    config_path=config_path,
                )
                for split, labels, scores in [
                    ("train", y.iloc[train], pt),
                    ("validation", y.iloc[valid], pv),
                ]:
                    point = select_constrained_recall(
                        operating_points(labels, scores), 0.55
                    )
                    rows.append(
                        {
                            "curve": curve,
                            "setting": setting,
                            "fold": number,
                            "split": split,
                            "ap": average_precision_score(labels, scores),
                            "roc_auc": roc_auc_score(labels, scores),
                            "log_loss": log_loss(labels, scores),
                            "constrained_recall": point["recall"] if point else np.nan,
                        }
                    )
                if curve == "depth":
                    predictions.setdefault(f"depth_{setting}", np.full(len(y), np.nan))[
                        valid
                    ] = pv
        for name, removed in GROUPS.items():
            columns = [c for c in X if c not in removed]
            _, pv, _ = fit_fold(
                X,
                y,
                train,
                valid,
                params,
                weight,
                42,
                columns,
                config_path=config_path,
            )
            ablations.append(
                {
                    "group": name,
                    "fold": number,
                    "ap": average_precision_score(y.iloc[valid], pv),
                    "constrained_recall": (
                        select_constrained_recall(
                            operating_points(y.iloc[valid], pv), 0.55
                        )
                        or {}
                    ).get("recall", np.nan),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(ablations), predictions


def plots(y, raw, p, threshold, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, scores in [("raw", raw), ("sigmoid", p)]:
        axes[0].hist(
            scores[y == 0],
            bins=30,
            alpha=0.35,
            density=True,
            label=f"{label} non-churn",
        )
        axes[0].hist(
            scores[y == 1], bins=30, alpha=0.35, density=True, label=f"{label} churn"
        )
    axes[0].axvline(threshold, color="black", linestyle="--")
    axes[0].set(xlabel="Predicted probability", ylabel="Density")
    axes[0].legend(fontsize=7)
    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    points = (
        pd.DataFrame({"bin": bins, "p": p, "y": y})
        .groupby("bin")
        .agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    )
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].scatter(points.predicted, points.observed, s=points.n / 8)
    axes[1].set(
        xlabel="Mean predicted churn", ylabel="Observed churn", xlim=(0, 1), ylim=(0, 1)
    )
    fig.tight_layout()
    fig.savefig(out / "probability_overlap_reliability.png", dpi=140)
    plt.close(fig)
    return points


def run(config_path="params.yaml"):
    config = load_config(config_path)
    path = Path(config["data"]["development_path"])
    data = pd.read_parquet(path)
    X, y_series = split_features_target(data, config_path)
    folds = make_folds(y_series, 5, config["project"]["random_state"])
    selection = json.loads((SOURCE / "revised_selection.json").read_text())
    cal = pd.read_csv(SOURCE / "calibration_oof.csv")
    raw_oof = pd.read_csv(SOURCE / "extended_trial_35_oof.csv")
    y, fold_ids = verify_alignment(data, selection, cal, raw_oof, folds, path)
    p, raw = cal.sigmoid_probability.to_numpy(), cal.raw_probability.to_numpy()
    threshold = selection["threshold"]
    baseline = binary_metrics(y, p, threshold)
    if tuple(int(baseline[k]) for k in ("tn", "fp", "fn", "tp")) != tuple(
        int(selection["selected_metrics"][k]) for k in ("tn", "fp", "fn", "tp")
    ):
        raise ValueError("Frozen confusion matrix was not reproduced")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seg = segments(X.reset_index(drop=True), y, p, threshold)
    seg.to_csv(OUTPUT / "segments.csv", index=False)
    group = np.select(
        [
            (y == 1) & (p >= threshold),
            (y == 0) & (p < threshold),
            (y == 0) & (p >= threshold),
        ],
        ["TP", "TN", "FP"],
        default="FN",
    )
    numeric = (
        pd.DataFrame(
            {
                "group": group,
                **{
                    c: X[c].to_numpy()
                    for c in ["tenure", "MonthlyCharges", "TotalCharges"]
                },
            }
        )
        .groupby("group")
        .agg(["mean", "median"])
    )
    numeric.to_csv(OUTPUT / "error_group_numeric.csv")
    proportions = []
    for column in [
        "Contract",
        "PaymentMethod",
        "InternetService",
        "TechSupport",
        "OnlineSecurity",
        "SeniorCitizen",
        "gender",
        "Partner",
        "Dependents",
    ]:
        table = pd.crosstab(group, X[column], normalize="index")
        for g in table.index:
            for value in table.columns:
                proportions.append(
                    {
                        "feature": column,
                        "value": str(value),
                        "error_group": g,
                        "proportion": table.loc[g, value],
                    }
                )
    pd.DataFrame(proportions).to_csv(
        OUTPUT / "error_group_categorical.csv", index=False
    )
    points = plots(y, raw, p, threshold, OUTPUT)
    points.to_csv(OUTPUT / "reliability_deciles.csv")
    curve = operating_points(y, p)
    curve.to_csv(OUTPUT / "precision_recall_curve.csv", index=False)
    constraints = {
        str(floor): select_constrained_recall(curve, floor)
        for floor in [0.50, 0.55, 0.60, 0.65, 0.70]
    }
    top = []
    for fraction in [0.1, 0.2, 0.3, 0.4]:
        idx = np.argsort(-p, kind="stable")[: round(len(y) * fraction)]
        tp = int(y[idx].sum())
        top.append(
            {
                "fraction": fraction,
                "workload": len(idx),
                "churners_identified": tp,
                "false_alerts": len(idx) - tp,
                "precision": tp / len(idx),
                "recall": tp / y.sum(),
                "lift": (tp / len(idx)) / y.mean(),
            }
        )
    pd.DataFrame(top).to_csv(OUTPUT / "top_risk.csv", index=False)
    profile = X.copy()
    profile["observed_churn"] = y
    grouped = profile.groupby(list(X.columns), dropna=False).observed_churn.agg(
        ["size", "nunique"]
    )
    duplicate = {
        "duplicate_profile_groups": int((grouped["size"] > 1).sum()),
        "duplicate_rows": int(grouped.loc[grouped["size"] > 1, "size"].sum()),
        "contradictory_profile_groups": int((grouped["nunique"] > 1).sum()),
        "contradictory_rows": int(grouped.loc[grouped["nunique"] > 1, "size"].sum()),
    }
    rows, ablations, candidates = curve_experiments(
        X,
        y_series,
        folds,
        selection["parameters"],
        selection["scale_pos_weight"],
        config_path,
    )
    rows.to_csv(OUTPUT / "learning_capacity_curves.csv", index=False)
    ablations.to_csv(OUTPUT / "ablations.csv", index=False)
    comparison = []
    for name, scores in {"frozen": p, **candidates}.items():
        fold_ap = [
            average_precision_score(y[valid], scores[valid]) for _, valid in folds
        ]
        point = select_constrained_recall(operating_points(y, scores), 0.55)
        comparison.append(
            {
                "model": name,
                "mean_fold_ap": np.mean(fold_ap),
                "pooled_ap": average_precision_score(y, scores),
                "pooled_roc_auc": roc_auc_score(y, scores),
                "constrained_recall": point["recall"],
                "paired_ap_delta": np.mean(
                    np.array(fold_ap)
                    - np.array(
                        [
                            average_precision_score(y[valid], p[valid])
                            for _, valid in folds
                        ]
                    )
                ),
            }
        )
    pd.DataFrame(comparison).to_csv(OUTPUT / "candidate_comparison.csv", index=False)
    findings = {
        "integrity": {
            "development_sha256": _checksum(path),
            "fold_sha256": _fold_hash(folds),
            "oof_aligned": True,
        },
        "baseline": {
            **baseline,
            "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p),
            "mean_fold_ap": float(
                np.mean([average_precision_score(y[v], p[v]) for _, v in folds])
            ),
        },
        "raw": {
            "ap": average_precision_score(y, raw),
            "brier": brier_score_loss(y, raw),
            "log_loss": log_loss(y, raw),
        },
        "constraints": constraints,
        "duplicates": duplicate,
        "borderline_errors_0.05": int(
            ((group == "FP") | (group == "FN"))
            .__and__(np.abs(p - threshold) <= 0.05)
            .sum()
        ),
        "confident_fn_below_0.1": int(((group == "FN") & (p < 0.1)).sum()),
        "confident_fp_above_0.7": int(((group == "FP") & (p >= 0.7)).sum()),
    }
    (OUTPUT / "findings.json").write_text(
        json.dumps(json_safe(findings), indent=2, allow_nan=False)
    )
    (OUTPUT / "frozen_configuration.json").write_text(
        json.dumps(
            {
                "recommendation": "retain_existing",
                "source": "revised_selection.json",
                "configuration": selection,
            },
            indent=2,
        )
    )
    mlflow.set_tracking_uri(config["model_training"]["mlflow_tracking_uri"])
    mlflow.set_experiment("customer-retention-model-diagnostics")
    with mlflow.start_run(
        run_name="phase_6a_plus_plus",
        tags={"phase": "6A++", "development_sha256": _checksum(path)},
    ) as run:
        mlflow.log_metrics(
            {
                "pooled_ap": findings["baseline"]["average_precision"],
                "mean_fold_ap": findings["baseline"]["mean_fold_ap"],
                "constrained_recall": findings["baseline"]["recall"],
            }
        )
        mlflow.log_artifacts(str(OUTPUT))
        findings["mlflow_run_id"] = run.info.run_id
    (OUTPUT / "findings.json").write_text(
        json.dumps(json_safe(findings), indent=2, allow_nan=False)
    )
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    run(args.config)
