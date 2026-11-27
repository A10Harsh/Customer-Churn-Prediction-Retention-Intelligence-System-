"""PostgreSQL persistence models for customer inputs and immutable predictions."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from retention.database.validation import CATEGORIES, COLUMN_MAP


def _category_constraints():
    constraints = []
    for name, allowed in CATEGORIES.items():
        column = COLUMN_MAP.get(name, name.lower())
        choices = ", ".join("'" + choice.replace("'", "''") + "'" for choice in allowed)
        constraints.append(
            CheckConstraint(
                f"{column} IN ({choices})", name=f"ck_customers_{column}_category"
            )
        )
    return tuple(constraints)


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"
    __table_args__ = (
        UniqueConstraint("external_id", name="uq_customers_external_id"),
        *_category_constraints(),
        CheckConstraint("external_id <> ''", name="ck_customers_external_id_nonempty"),
        CheckConstraint("senior_citizen IN (0, 1)", name="ck_customers_senior_citizen"),
        CheckConstraint("tenure >= 0", name="ck_customers_tenure"),
        CheckConstraint("monthly_charges >= 0", name="ck_customers_monthly_charges"),
        CheckConstraint("total_charges >= 0", name="ck_customers_total_charges"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    senior_citizen: Mapped[int] = mapped_column(Integer, nullable=False)
    partner: Mapped[str] = mapped_column(String(3), nullable=False)
    dependents: Mapped[str] = mapped_column(String(3), nullable=False)
    tenure: Mapped[int] = mapped_column(Integer, nullable=False)
    phone_service: Mapped[str] = mapped_column(String(3), nullable=False)
    multiple_lines: Mapped[str] = mapped_column(String(32), nullable=False)
    internet_service: Mapped[str] = mapped_column(String(16), nullable=False)
    online_security: Mapped[str] = mapped_column(String(32), nullable=False)
    online_backup: Mapped[str] = mapped_column(String(32), nullable=False)
    device_protection: Mapped[str] = mapped_column(String(32), nullable=False)
    tech_support: Mapped[str] = mapped_column(String(32), nullable=False)
    streaming_tv: Mapped[str] = mapped_column(String(32), nullable=False)
    streaming_movies: Mapped[str] = mapped_column(String(32), nullable=False)
    contract: Mapped[str] = mapped_column(String(16), nullable=False)
    paperless_billing: Mapped[str] = mapped_column(String(3), nullable=False)
    payment_method: Mapped[str] = mapped_column(String(32), nullable=False)
    monthly_charges: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    total_charges: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="customer")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "churn_probability >= 0 AND churn_probability <= 1",
            name="ck_predictions_probability",
        ),
        CheckConstraint(
            "threshold >= 0 AND threshold <= 1", name="ck_predictions_threshold"
        ),
        CheckConstraint("predicted_churn IN (0, 1)", name="ck_predictions_class"),
        CheckConstraint(
            "jsonb_typeof(predictor_snapshot) = 'object'",
            name="ck_predictions_snapshot_object",
        ),
        CheckConstraint(
            "length(model_fingerprint) = 64", name="ck_predictions_fingerprint_length"
        ),
        CheckConstraint(
            "(predicted_churn = 1 AND churn_probability >= threshold) OR "
            "(predicted_churn = 0 AND churn_probability < threshold)",
            name="ck_predictions_class_threshold",
        ),
        Index("ix_predictions_customer_time", "customer_id", "predicted_at", "id"),
        Index("ix_predictions_predicted_at", "predicted_at"),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="RESTRICT"), nullable=False
    )
    churn_probability: Mapped[Decimal] = mapped_column(Numeric(22, 20), nullable=False)
    predicted_churn: Mapped[int] = mapped_column(Integer, nullable=False)
    threshold: Mapped[Decimal] = mapped_column(Numeric(22, 20), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    predictor_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    customer: Mapped[Customer] = relationship(back_populates="predictions")
