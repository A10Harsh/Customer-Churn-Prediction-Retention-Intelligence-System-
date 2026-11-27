"""Frozen model inference with atomic customer and prediction persistence."""

from functools import lru_cache
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from retention.database.connection import session_scope
from retention.database.repository import (
    get_customer,
    insert_prediction,
    upsert_customer,
)
from retention.database.validation import customer_predictors, validate_predictors
from retention.models.predictor import ChurnPredictor, load_predictor

ARTIFACT = Path("artifacts/final_model/predictor.joblib")
MODEL_NAME = "customer-churn-predictor"
MODEL_VERSION = "1"
FROZEN_THRESHOLD = 0.3304499866713542


@lru_cache(maxsize=1)
def frozen_predictor() -> ChurnPredictor:
    predictor = load_predictor(ARTIFACT)
    if predictor.threshold != FROZEN_THRESHOLD or not predictor.metadata.get(
        "configuration_sha256"
    ):
        raise ValueError("Frozen predictor metadata mismatch")
    return predictor


def predict_and_persist(
    factory: sessionmaker[Session],
    external_id: str,
    predictors: dict | None = None,
    *,
    predictor: ChurnPredictor | None = None,
) -> tuple[str, dict]:
    """Return a persisted UUID and inference result after the transaction commits."""
    model = predictor if predictor is not None else frozen_predictor()
    if model.threshold != FROZEN_THRESHOLD:
        raise ValueError("Unexpected model threshold")
    with session_scope(factory) as session:
        if predictors is None:
            customer = get_customer(session, external_id)
            if customer is None:
                raise LookupError("Customer not found")
            predictors = customer_predictors(customer)
        normalized = validate_predictors(predictors)
        if set(model.columns) != set(normalized):
            raise ValueError("Predictor schema mismatch")
        result = model.predict(normalized).iloc[0]
        if float(result["threshold"]) != FROZEN_THRESHOLD:
            raise ValueError("Inference threshold differs from frozen threshold")
        customer = upsert_customer(session, external_id, normalized)
        prediction = insert_prediction(
            session,
            customer=customer,
            probability=float(result["churn_probability"]),
            predicted_churn=int(result["predicted_churn"]),
            threshold=float(result["threshold"]),
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            model_fingerprint=model.metadata["configuration_sha256"],
            snapshot=normalized,
        )
        prediction_id = str(prediction.id)
        output = {
            "churn_probability": float(result["churn_probability"]),
            "predicted_churn": int(result["predicted_churn"]),
            "threshold": float(result["threshold"]),
        }
    return prediction_id, output
