"""Add immutable explanations for existing prediction records.

Revision ID: 0002_explanations
Revises: 0001_customers_predictions
"""

from alembic import op

revision = "0002_explanations"
down_revision = "0001_customers_predictions"
branch_labels = None
depends_on = None


# Frozen PostgreSQL DDL reviewed against the Phase 8 SQLAlchemy metadata.
EXPLANATIONS_DDL = """CREATE TABLE explanations (
    id UUID NOT NULL,
    prediction_id UUID NOT NULL,
    explanation_method VARCHAR(32) NOT NULL,
    explanation_version VARCHAR(16) NOT NULL,
    top_k INTEGER NOT NULL,
    explained_output_space VARCHAR(32) NOT NULL,
    base_value FLOAT(53) NOT NULL,
    raw_margin FLOAT(53) NOT NULL,
    raw_probability NUMERIC(22, 20) NOT NULL,
    calibrated_probability NUMERIC(22, 20) NOT NULL,
    additivity_residual FLOAT(53) NOT NULL,
    full_payload JSONB NOT NULL,
    original_feature_contributions JSONB NOT NULL,
    top_risk_increasing JSONB NOT NULL,
    top_risk_decreasing JSONB NOT NULL,
    model_fingerprint VARCHAR(64) NOT NULL,
    explainer_fingerprint VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT uq_explanations_request UNIQUE (prediction_id, explanation_method,
        explanation_version, explainer_fingerprint, top_k),
    CONSTRAINT ck_explanations_top_k CHECK (top_k BETWEEN 1 AND 19),
    CONSTRAINT ck_explanations_output_space CHECK (explained_output_space =
        'raw_xgboost_margin'),
    CONSTRAINT ck_explanations_payload_object CHECK (jsonb_typeof(full_payload) =
        'object'),
    CONSTRAINT ck_explanations_contributions_array CHECK
        (jsonb_typeof(original_feature_contributions) = 'array'),
    CONSTRAINT ck_explanations_factors_array CHECK (jsonb_typeof(top_risk_increasing)
        = 'array' AND jsonb_typeof(top_risk_decreasing) = 'array'),
    CONSTRAINT ck_explanations_probabilities CHECK (raw_probability >= 0 AND
        raw_probability <= 1 AND calibrated_probability >= 0 AND
        calibrated_probability <= 1),
    CONSTRAINT ck_explanations_fingerprints CHECK (length(model_fingerprint) = 64 AND
        length(explainer_fingerprint) = 64),
    FOREIGN KEY(prediction_id) REFERENCES predictions (id) ON DELETE RESTRICT
)"""


def upgrade():
    op.execute(EXPLANATIONS_DDL)


def downgrade():
    op.drop_table("explanations")
