"""Safe, type-aware exploratory association measures."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def numerical_correlations(
    df: pd.DataFrame, features: list[str]
) -> dict[str, pd.DataFrame]:
    """Return Pearson, Spearman, and tied-rank Kendall Tau-b matrices."""
    data = df[features].apply(pd.to_numeric, errors="coerce")
    return {
        name: data.corr(method=method)
        for name, method in (
            ("pearson", "pearson"),
            ("spearman", "spearman"),
            ("kendall_tau_b", "kendall"),
        )
    }


def point_biserial_with_target(
    df: pd.DataFrame, features: list[str], target: str, positive: str = "Yes"
) -> pd.DataFrame:
    """Calculate signed Pearson/point-biserial correlation with a binary target."""
    binary = df[target].eq(positive).astype("int8")
    return pd.DataFrame(
        {
            "point_biserial_r": {
                f: pd.to_numeric(df[f], errors="coerce").corr(binary) for f in features
            }
        }
    ).sort_values("point_biserial_r", key=lambda x: x.abs(), ascending=False)


def cramers_v(x: pd.Series, y: pd.Series, bias_corrected: bool = True) -> float:
    """Calculate conventional or Bergsma-style bias-corrected Cramer's V."""
    table = pd.crosstab(x, y, dropna=False).to_numpy(dtype=float)
    n = table.sum()
    rows, columns = table.shape
    if n <= 1 or rows < 2 or columns < 2:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    valid = expected > 0
    if not valid.any():
        return 0.0
    phi_squared = np.sum(((table - expected) ** 2)[valid] / expected[valid]) / n
    if not bias_corrected:
        denominator = min(rows - 1, columns - 1)
        return float(np.sqrt(phi_squared / denominator)) if denominator else 0.0
    corrected_phi = max(0.0, phi_squared - (rows - 1) * (columns - 1) / (n - 1))
    corrected_rows = rows - (rows - 1) ** 2 / (n - 1)
    corrected_columns = columns - (columns - 1) ** 2 / (n - 1)
    denominator = min(corrected_rows - 1, corrected_columns - 1)
    return float(np.sqrt(corrected_phi / denominator)) if denominator > 0 else 0.0


def categorical_target_associations(
    df: pd.DataFrame, features: list[str], target: str
) -> pd.DataFrame:
    """Rank categorical predictors by corrected and conventional Cramer's V."""
    return (
        pd.DataFrame(
            {
                "feature": features,
                "bias_corrected_v": [cramers_v(df[f], df[target]) for f in features],
                "conventional_v": [
                    cramers_v(df[f], df[target], False) for f in features
                ],
            }
        )
        .set_index("feature")
        .sort_values("bias_corrected_v", ascending=False)
    )


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Calculate eta, returning zero for empty or zero-variance inputs."""
    codes, _ = pd.factorize(categories, sort=False)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = (codes >= 0) & np.isfinite(numeric)
    codes, numeric = codes[valid], numeric[valid]
    if not numeric.size:
        return 0.0
    total = np.sum((numeric - numeric.mean()) ** 2)
    if total <= 0:
        return 0.0
    between = sum(
        group.size * (group.mean() - numeric.mean()) ** 2
        for code in np.unique(codes)
        if (group := numeric[codes == code]).size
    )
    return float(np.sqrt(max(0.0, between / total)))


def correlation_ratio_matrix(
    df: pd.DataFrame, categorical: list[str], numerical: list[str]
) -> pd.DataFrame:
    """Return eta for each categorical-to-numerical pairing."""
    return pd.DataFrame(
        {
            n: {c: correlation_ratio(df[c], df[n]) for c in categorical}
            for n in numerical
        }
    )


def mutual_information_ranking(
    df: pd.DataFrame,
    numerical: list[str],
    categorical: list[str],
    target: str,
    random_state: int,
    n_neighbors: int,
) -> pd.DataFrame:
    """Estimate deterministic marginal mutual information in nats."""
    features = numerical + categorical
    encoded = pd.DataFrame(index=df.index)
    for feature in features:
        encoded[feature] = (
            df[feature].astype("category").cat.codes
            if feature in categorical
            else df[feature]
        )
    discrete = np.array([f == "tenure" or f in categorical for f in features])
    values = mutual_info_classif(
        encoded,
        df[target].eq("Yes").astype("int8"),
        discrete_features=discrete,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return pd.DataFrame(
        {"mutual_information_nats": values}, index=features
    ).sort_values("mutual_information_nats", ascending=False)


def strong_associations(
    correlations: dict[str, pd.DataFrame],
    categorical: pd.DataFrame,
    point_biserial: pd.DataFrame,
    numerical_threshold: float,
    categorical_threshold: float,
    target_threshold: float,
) -> pd.DataFrame:
    """Screen strong associations without making feature-selection decisions."""
    rows: list[dict[str, object]] = []
    for measure in ("pearson", "spearman"):
        matrix = correlations[measure]
        for i, left in enumerate(matrix):
            for right in matrix.columns[i + 1 :]:
                value = matrix.loc[left, right]
                if pd.notna(value) and abs(value) >= numerical_threshold:
                    rows.append(
                        {
                            "feature_1": left,
                            "feature_2": right,
                            "measure": measure,
                            "value": value,
                        }
                    )
    for feature, row in categorical.iterrows():
        if row["bias_corrected_v"] >= categorical_threshold:
            rows.append(
                {
                    "feature_1": feature,
                    "feature_2": "Churn",
                    "measure": "cramers_v",
                    "value": row["bias_corrected_v"],
                }
            )
    for feature, row in point_biserial.iterrows():
        if (
            pd.notna(row["point_biserial_r"])
            and abs(row["point_biserial_r"]) >= target_threshold
        ):
            rows.append(
                {
                    "feature_1": feature,
                    "feature_2": "Churn",
                    "measure": "point_biserial",
                    "value": row["point_biserial_r"],
                }
            )
    return pd.DataFrame(rows, columns=["feature_1", "feature_2", "measure", "value"])
