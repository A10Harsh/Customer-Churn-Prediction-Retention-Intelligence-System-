"""Diagnostic-only univariate outlier and consistency screening."""

import numpy as np
import pandas as pd


def iqr_flags(
    data: pd.DataFrame, features: list[str], multiplier: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return IQR fence summaries and row-level flags without altering values."""
    flags = pd.DataFrame(False, index=data.index, columns=features)
    rows = []
    for feature in features:
        q1, q3 = data[feature].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_fence, upper_fence = q1 - multiplier * iqr, q3 + multiplier * iqr
        lower, upper = data[feature].lt(lower_fence), data[feature].gt(upper_fence)
        flags[feature] = lower | upper
        rows.append(
            {
                "feature": feature,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence": lower_fence,
                "upper_fence": upper_fence,
                "lower_count": int(lower.sum()),
                "upper_count": int(upper.sum()),
                "flagged_count": int((lower | upper).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("feature"), flags


def standard_z_scores(values: pd.Series) -> pd.Series:
    """Calculate population Z-scores, safely returning zeros at zero variance."""
    standard_deviation = values.std(ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / standard_deviation


def modified_z_scores(values: pd.Series) -> tuple[pd.Series, float, float]:
    """Calculate median/MAD scores, safely returning zeros at zero MAD."""
    median = values.median()
    mad = (values - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(0.0, index=values.index), float(median), float(mad)
    return 0.67449 * (values - median) / mad, float(median), float(mad)


def score_flags(
    data: pd.DataFrame,
    features: list[str],
    z_threshold: float,
    modified_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return Z and modified-Z flags plus feature-level summaries."""
    z = data[features].apply(standard_z_scores)
    modified = pd.DataFrame(index=data.index)
    rows = []
    for feature in features:
        modified[feature], median, mad = modified_z_scores(data[feature])
        rows.append(
            {
                "feature": feature,
                "median": median,
                "mad": mad,
                "z_flagged_count": int(z[feature].abs().gt(z_threshold).sum()),
                "modified_z_flagged_count": int(
                    modified[feature].abs().gt(modified_threshold).sum()
                ),
            }
        )
    return (
        z.abs().gt(z_threshold),
        modified.abs().gt(modified_threshold),
        pd.DataFrame(rows).set_index("feature"),
    )


def percentile_flags(
    data: pd.DataFrame,
    features: list[str],
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return strict tail flags and observed percentile thresholds."""
    flags = pd.DataFrame(False, index=data.index, columns=features)
    rows = []
    for feature in features:
        lower, upper = data[feature].quantile([lower_quantile, upper_quantile])
        flags[feature] = data[feature].lt(lower) | data[feature].gt(upper)
        rows.append(
            {
                "feature": feature,
                "lower_threshold": lower,
                "upper_threshold": upper,
                "flagged_count": int(flags[feature].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("feature"), flags


def consistency_summary(df: pd.DataFrame) -> dict[str, int]:
    """Summarize zero-tenure, billing, phone, and internet-service consistency."""
    addons = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    phone = (
        (df["PhoneService"] == "No") & (df["MultipleLines"] != "No phone service")
    ) | ((df["PhoneService"] == "Yes") & (df["MultipleLines"] == "No phone service"))
    internet = pd.Series(False, index=df.index)
    for feature in addons:
        internet |= (
            (df["InternetService"] == "No") & (df[feature] != "No internet service")
        ) | ((df["InternetService"] != "No") & (df[feature] == "No internet service"))
    return {
        "zero_tenure": int(df["tenure"].eq(0).sum()),
        "zero_total_charges": int(df["TotalCharges"].eq(0).sum()),
        "zero_tenure_and_zero_total": int(
            (df["tenure"].eq(0) & df["TotalCharges"].eq(0)).sum()
        ),
        "zero_tenure_nonzero_total": int(
            (df["tenure"].eq(0) & df["TotalCharges"].ne(0)).sum()
        ),
        "positive_tenure_zero_total": int(
            (df["tenure"].gt(0) & df["TotalCharges"].eq(0)).sum()
        ),
        "nonpositive_monthly_charge": int(df["MonthlyCharges"].le(0).sum()),
        "negative_total_charge": int(df["TotalCharges"].lt(0).sum()),
        "phone_service_state_inconsistency": int(phone.sum()),
        "internet_service_state_inconsistency": int(internet.sum()),
    }


def outlier_screening(
    df: pd.DataFrame,
    features: list[str],
    iqr_multiplier: float,
    z_threshold: float,
    modified_threshold: float,
    percentiles: tuple[float, float],
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    """Run all recurring diagnostic screens and return summaries and flags."""
    iqr_table, iqr = iqr_flags(df, features, iqr_multiplier)
    z, modified, score_table = score_flags(
        df, features, z_threshold, modified_threshold
    )
    percentile_table, percentile = percentile_flags(df, features, *percentiles)
    methods = pd.DataFrame(
        {
            "IQR": iqr.any(axis=1),
            "Z-score": z.any(axis=1),
            "Modified Z-score": modified.any(axis=1),
            "Percentile": percentile.any(axis=1),
        }
    )
    return (
        {
            "method_flagged_rows": {c: int(methods[c].sum()) for c in methods},
            "rows_flagged_by_any_method": int(methods.any(axis=1).sum()),
            "consistency": consistency_summary(df),
            "note": (
                "Diagnostic flags only; no observations were removed, clipped, "
                "or transformed."
            ),
        },
        {
            "iqr": iqr_table,
            "score": score_table,
            "percentile": percentile_table,
            "method_flags": methods,
        },
    )
