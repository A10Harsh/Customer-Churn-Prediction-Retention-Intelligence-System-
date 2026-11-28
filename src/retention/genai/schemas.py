"""Only these fields may be generated or rendered."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ActionID(StrEnum):
    NO_ACTION = "NO_ACTION"
    ONBOARDING_SUPPORT = "ONBOARDING_SUPPORT"
    CONTRACT_REVIEW = "CONTRACT_REVIEW"
    TECH_SUPPORT_REVIEW = "TECH_SUPPORT_REVIEW"
    SERVICE_QUALITY_REVIEW = "SERVICE_QUALITY_REVIEW"
    BILLING_REVIEW = "BILLING_REVIEW"
    PLAN_VALUE_REVIEW = "PLAN_VALUE_REVIEW"
    SECURITY_SERVICE_REVIEW = "SECURITY_SERVICE_REVIEW"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class Priority(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class Channel(StrEnum):
    none = "none"
    phone = "phone"
    email = "email"
    internal = "internal"


class ReasonCode(StrEnum):
    """Approved reason codes that the LLM may select."""

    EARLY_TENURE = "EARLY_TENURE"
    MONTH_TO_MONTH = "MONTH_TO_MONTH"
    NO_TECH_SUPPORT = "NO_TECH_SUPPORT"
    NO_ONLINE_SECURITY = "NO_ONLINE_SECURITY"
    HIGH_MONTHLY_CHARGES = "HIGH_MONTHLY_CHARGES"
    SERVICE_CUSTOMER = "SERVICE_CUSTOMER"
    SHAP_RISK_FACTOR = "SHAP_RISK_FACTOR"
    PREDICTED_CHURN_RISK = "PREDICTED_CHURN_RISK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    RETAINED_CLASS = "RETAINED_CLASS"


# --- Constrained LLM output: selection only, no free text ---


class LLMSelection(BaseModel):
    """Strict schema for LLM output: action selection with evidence IDs only."""

    model_config = ConfigDict(extra="forbid")
    action_id: ActionID
    evidence_ids: list[str] = Field(min_length=1, max_length=6)
    reason_codes: list[ReasonCode] = Field(min_length=1, max_length=4)
    human_review_required: bool


# --- Full recommendation (populated by deterministic renderer) ---


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: ActionID
    priority: Priority
    channel: Channel
    summary: str = Field(min_length=5, max_length=240)
    rationale: str = Field(min_length=5, max_length=500)
    evidence_refs: list[str] = Field(min_length=1, max_length=6)
    talking_points: list[str] = Field(max_length=4)
    human_approval_required: bool
    limitations: str = Field(min_length=5, max_length=300)
    reason_codes: list[ReasonCode] = Field(default_factory=list, max_length=4)
    rendering_version: str | None = None
