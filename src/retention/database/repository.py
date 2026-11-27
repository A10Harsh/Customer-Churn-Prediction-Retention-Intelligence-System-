"""Small SQLAlchemy repositories; caller owns transaction boundaries."""

from decimal import Decimal, InvalidOperation
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from retention.database.models import Customer, Prediction
from retention.database.validation import model_values, validate_predictors


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError("A nonempty external customer identifier is required")
    return value


def _page(limit: int, offset: int) -> None:
    if (
        isinstance(limit, bool)
        or isinstance(offset, bool)
        or not isinstance(limit, int)
        or not isinstance(offset, int)
        or not 1 <= limit <= 100
        or offset < 0
    ):
        raise ValueError("limit must be 1..100 and offset must be nonnegative")


def get_customer(session: Session, external_id: str) -> Customer | None:
    return session.scalar(
        select(Customer).where(Customer.external_id == _identifier(external_id))
    )


def upsert_customer(session: Session, external_id: str, predictors: dict) -> Customer:
    values = model_values(validate_predictors(predictors))
    statement = insert(Customer).values(external_id=_identifier(external_id), **values)
    statement = statement.on_conflict_do_update(
        constraint="uq_customers_external_id",
        set_={**values, "updated_at": func.now()},
    ).returning(Customer)
    return session.scalars(statement.execution_options(populate_existing=True)).one()


def update_customer(
    session: Session, external_id: str, predictors: dict
) -> Customer | None:
    customer = get_customer(session, external_id)
    if customer is None:
        return None
    for key, value in model_values(validate_predictors(predictors)).items():
        setattr(customer, key, value)
    session.flush()
    return customer


def list_customers(
    session: Session, *, limit: int = 50, offset: int = 0
) -> list[Customer]:
    _page(limit, offset)
    return list(
        session.scalars(
            select(Customer).order_by(Customer.id).limit(limit).offset(offset)
        )
    )


def _unit_decimal(value: object, name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not number.is_finite() or not 0 <= number <= 1:
        raise ValueError(f"Invalid {name}")
    if number.as_tuple().exponent < -20:
        raise ValueError(f"{name} exceeds database precision")
    return number


def insert_prediction(
    session: Session,
    *,
    customer: Customer,
    probability: float,
    predicted_churn: int,
    threshold: float,
    model_name: str,
    model_version: str,
    model_fingerprint: str,
    snapshot: dict,
) -> Prediction:
    p = _unit_decimal(probability, "probability")
    t = _unit_decimal(threshold, "threshold")
    if type(predicted_churn) is not int or predicted_churn != int(p >= t):
        raise ValueError("Predicted class must match probability and threshold")
    if (
        not model_name
        or not model_version
        or len(model_name) > 128
        or len(model_version) > 32
    ):
        raise ValueError("Model name and version are required")
    if len(model_fingerprint) != 64 or any(
        ch not in "0123456789abcdef" for ch in model_fingerprint
    ):
        raise ValueError("Expected a SHA-256 model fingerprint")
    normalized = validate_predictors(snapshot)
    if customer.id is None:
        session.flush()
    prediction = Prediction(
        customer_id=customer.id,
        churn_probability=p,
        predicted_churn=predicted_churn,
        threshold=t,
        model_name=model_name,
        model_version=model_version,
        model_fingerprint=model_fingerprint,
        predictor_snapshot={
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in normalized.items()
        },
    )
    session.add(prediction)
    session.flush()
    return prediction


def get_prediction(session: Session, prediction_id: UUID | str) -> Prediction | None:
    try:
        parsed = UUID(str(prediction_id))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid prediction identifier") from exc
    return session.get(Prediction, parsed)


def customer_history(
    session: Session, external_id: str, *, limit: int = 50, offset: int = 0
) -> list[Prediction]:
    _page(limit, offset)
    return list(
        session.scalars(
            select(Prediction)
            .join(Customer)
            .where(Customer.external_id == _identifier(external_id))
            .order_by(Prediction.predicted_at.desc(), Prediction.id.desc())
            .limit(limit)
            .offset(offset)
        )
    )


def latest_prediction(session: Session, external_id: str) -> Prediction | None:
    history = customer_history(session, external_id, limit=1)
    return history[0] if history else None


def list_predictions(
    session: Session, *, limit: int = 50, offset: int = 0
) -> list[Prediction]:
    _page(limit, offset)
    return list(
        session.scalars(
            select(Prediction)
            .order_by(Prediction.predicted_at.desc(), Prediction.id.desc())
            .limit(limit)
            .offset(offset)
        )
    )
