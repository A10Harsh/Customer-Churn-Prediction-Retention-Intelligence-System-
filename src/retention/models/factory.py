"""Initial classical estimators and their fold-local preprocessing pipelines."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from retention.features.preprocessing import make_preprocessor


def make_model(
    model: str,
    strategy: str,
    seed: int,
    parameters: dict,
    scale_pos_weight: float | None = None,
) -> object:
    """Instantiate one estimator; XGBoost's weight is computed per fold."""
    if model == "dummy":
        if strategy != "unweighted":
            raise ValueError("DummyClassifier only supports unweighted strategy.")
        return DummyClassifier(strategy="prior")
    if strategy not in {"unweighted", "weighted"}:
        raise ValueError("Unknown imbalance strategy.")
    if model == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced" if strategy == "weighted" else None,
            random_state=seed,
            **parameters,
        )
    if model == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced" if strategy == "weighted" else None,
            random_state=seed,
            **parameters,
        )
    if model == "xgboost":
        if strategy == "weighted" and (
            scale_pos_weight is None or scale_pos_weight <= 0
        ):
            raise ValueError("Weighted XGBoost requires a positive fold weight.")
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            scale_pos_weight=scale_pos_weight if strategy == "weighted" else 1.0,
            **parameters,
        )
    raise ValueError(f"Unknown model: {model}")


def make_estimator_pipeline(
    model: str,
    feature_mode: str,
    strategy: str,
    seed: int,
    parameters: dict,
    scale_pos_weight: float | None = None,
    config_path: str = "params.yaml",
) -> Pipeline:
    """Return a fresh, unfitted preprocessing-plus-estimator pipeline."""
    family = "linear" if model == "logistic_regression" else "tree"
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(feature_mode, family, config_path=config_path),
            ),
            (
                "estimator",
                make_model(model, strategy, seed, parameters, scale_pos_weight),
            ),
        ]
    )
