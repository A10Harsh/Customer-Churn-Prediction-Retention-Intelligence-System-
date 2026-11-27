"""Create customers and immutable prediction history.

Revision ID: 0001_customers_predictions
Revises:
"""

from alembic import op

revision = "0001_customers_predictions"
down_revision = None
branch_labels = None
depends_on = None


# Frozen PostgreSQL DDL reviewed from the Phase 7 SQLAlchemy metadata.
# Keep this revision independent of later ORM model changes.
CUSTOMERS_DDL = """CREATE TABLE customers (
    id BIGSERIAL NOT NULL,
    external_id VARCHAR(128) NOT NULL,
    gender VARCHAR(16) NOT NULL,
    senior_citizen INTEGER NOT NULL,
    partner VARCHAR(3) NOT NULL,
    dependents VARCHAR(3) NOT NULL,
    tenure INTEGER NOT NULL,
    phone_service VARCHAR(3) NOT NULL,
    multiple_lines VARCHAR(32) NOT NULL,
    internet_service VARCHAR(16) NOT NULL,
    online_security VARCHAR(32) NOT NULL,
    online_backup VARCHAR(32) NOT NULL,
    device_protection VARCHAR(32) NOT NULL,
    tech_support VARCHAR(32) NOT NULL,
    streaming_tv VARCHAR(32) NOT NULL,
    streaming_movies VARCHAR(32) NOT NULL,
    contract VARCHAR(16) NOT NULL,
    paperless_billing VARCHAR(3) NOT NULL,
    payment_method VARCHAR(32) NOT NULL,
    monthly_charges NUMERIC(12, 4) NOT NULL,
    total_charges NUMERIC(14, 4) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uq_customers_external_id UNIQUE (external_id),
    CONSTRAINT ck_customers_gender_category CHECK (gender IN ('Male', 'Female')),
    CONSTRAINT ck_customers_partner_category CHECK (partner IN ('Yes', 'No')),
    CONSTRAINT ck_customers_dependents_category CHECK (dependents IN ('Yes', 'No')),
    CONSTRAINT ck_customers_phone_service_category CHECK (phone_service IN ('Yes',
        'No')),
    CONSTRAINT ck_customers_multiple_lines_category CHECK (multiple_lines IN ('Yes',
        'No', 'No phone service')),
    CONSTRAINT ck_customers_internet_service_category CHECK (internet_service IN
        ('DSL', 'Fiber optic', 'No')),
    CONSTRAINT ck_customers_online_security_category CHECK (online_security IN
        ('Yes', 'No', 'No internet service')),
    CONSTRAINT ck_customers_online_backup_category CHECK (online_backup IN ('Yes',
        'No', 'No internet service')),
    CONSTRAINT ck_customers_device_protection_category CHECK (device_protection IN
        ('Yes', 'No', 'No internet service')),
    CONSTRAINT ck_customers_tech_support_category CHECK (tech_support IN ('Yes',
        'No', 'No internet service')),
    CONSTRAINT ck_customers_streaming_tv_category CHECK (streaming_tv IN ('Yes',
        'No', 'No internet service')),
    CONSTRAINT ck_customers_streaming_movies_category CHECK (streaming_movies IN
        ('Yes', 'No', 'No internet service')),
    CONSTRAINT ck_customers_contract_category CHECK (contract IN ('Month-to-month',
        'One year', 'Two year')),
    CONSTRAINT ck_customers_paperless_billing_category CHECK (paperless_billing IN
        ('Yes', 'No')),
    CONSTRAINT ck_customers_payment_method_category CHECK (payment_method IN
        ('Electronic check', 'Mailed check', 'Bank transfer (automatic)',
        'Credit card (automatic)')),
    CONSTRAINT ck_customers_external_id_nonempty CHECK (external_id <> ''),
    CONSTRAINT ck_customers_senior_citizen CHECK (senior_citizen IN (0, 1)),
    CONSTRAINT ck_customers_tenure CHECK (tenure >= 0),
    CONSTRAINT ck_customers_monthly_charges CHECK (monthly_charges >= 0),
    CONSTRAINT ck_customers_total_charges CHECK (total_charges >= 0)
)"""

PREDICTIONS_DDL = """CREATE TABLE predictions (
    id UUID NOT NULL,
    customer_id BIGINT NOT NULL,
    churn_probability NUMERIC(22, 20) NOT NULL,
    predicted_churn INTEGER NOT NULL,
    threshold NUMERIC(22, 20) NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    model_fingerprint VARCHAR(64) NOT NULL,
    predictor_snapshot JSONB NOT NULL,
    predicted_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT ck_predictions_probability CHECK (churn_probability >= 0 AND
        churn_probability <= 1),
    CONSTRAINT ck_predictions_threshold CHECK (threshold >= 0 AND threshold <= 1),
    CONSTRAINT ck_predictions_class CHECK (predicted_churn IN (0, 1)),
    CONSTRAINT ck_predictions_snapshot_object CHECK (jsonb_typeof(predictor_snapshot)
        = 'object'),
    CONSTRAINT ck_predictions_fingerprint_length CHECK (length(model_fingerprint) =
        64),
    CONSTRAINT ck_predictions_class_threshold CHECK ((predicted_churn = 1 AND
        churn_probability >= threshold) OR (predicted_churn = 0 AND churn_probability
        < threshold)),
    FOREIGN KEY(customer_id) REFERENCES customers (id) ON DELETE RESTRICT
)"""


def upgrade():
    op.execute(CUSTOMERS_DDL)
    op.execute(PREDICTIONS_DDL)
    op.create_index(
        "ix_predictions_customer_time",
        "predictions",
        ["customer_id", "predicted_at", "id"],
    )
    op.create_index("ix_predictions_predicted_at", "predictions", ["predicted_at"])


def downgrade():
    op.drop_table("predictions")
    op.drop_table("customers")
