"""Persist policy-checked next-best-action recommendations.

Revision ID: 0003_recommendations
Revises: 0002_explanations
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0003_recommendations"
down_revision = "0002_explanations"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "recommendations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "prediction_id",
            UUID(as_uuid=True),
            sa.ForeignKey("predictions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "explanation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("explanations.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("action_id", sa.String(32), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(32), nullable=False),
        sa.Column("policy_version", sa.String(32), nullable=False),
        sa.Column("generation_source", sa.String(16), nullable=False),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("attempt", sa.Integer, nullable=False),
        sa.Column("latency_ms", sa.Float(53), nullable=False),
        sa.Column("input_tokens", sa.Integer),
        sa.Column("output_tokens", sa.Integer),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "prediction_id",
            "request_fingerprint",
            "attempt",
            name="uq_recommendations_request_attempt",
        ),
        sa.CheckConstraint("attempt >= 1", name="ck_recommendations_attempt"),
        sa.CheckConstraint("latency_ms >= 0", name="ck_recommendations_latency"),
        sa.CheckConstraint(
            "generation_source IN ('llm', 'rules', 'fallback')",
            name="ck_recommendations_source",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(payload) = 'object'", name="ck_recommendations_payload"
        ),
    )
    op.create_index(
        "ix_recommendations_prediction_time",
        "recommendations",
        ["prediction_id", "created_at"],
    )
    # Composite FK makes cross-prediction explanation references impossible.
    op.create_unique_constraint(
        "uq_explanations_id_prediction", "explanations", ["id", "prediction_id"]
    )
    op.create_foreign_key(
        "fk_recommendations_matching_explanation",
        "recommendations",
        "explanations",
        ["explanation_id", "prediction_id"],
        ["id", "prediction_id"],
        ondelete="RESTRICT",
    )


def downgrade():
    op.drop_table("recommendations")
    op.drop_constraint("uq_explanations_id_prediction", "explanations", type_="unique")
