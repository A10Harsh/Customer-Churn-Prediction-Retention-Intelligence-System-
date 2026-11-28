"""Deterministic checks over untrusted generated text and fields.

Applies identical validation to LLM, deterministic and fallback
recommendations.  Treats customer context as untrusted data.
"""

import re

from retention.genai.schemas import LLMSelection, ReasonCode, Recommendation

FORBIDDEN = re.compile(
    r"\b(discount|coupon|free month|save \$|caused|"
    r"because of your age|because of your gender|complain(?:ed|t)?)\b",
    re.I,
)
GUARANTEE = re.compile(r"\bguarantees?\b", re.I)
NEGATED_GUARANTEE = re.compile(r"\b(?:not|no|never)\s+guarantees?\b", re.I)

DEMOGRAPHIC_ATTRIBUTES = {"gender", "SeniorCitizen", "senior_citizen", "age", "race"}

# Maps action IDs to the evidence-ref prefixes/IDs that validly support them.
ACTION_EVIDENCE_RULES: dict[str, set[str]] = {
    "ONBOARDING_SUPPORT": {"attr:tenure", "shap:tenure", "prediction:risk"},
    "CONTRACT_REVIEW": {"attr:Contract", "shap:Contract", "prediction:risk"},
    "TECH_SUPPORT_REVIEW": {"attr:TechSupport", "shap:TechSupport", "prediction:risk"},
    "SERVICE_QUALITY_REVIEW": {
        "attr:InternetService",
        "shap:InternetService",
        "attr:StreamingTV",
        "shap:StreamingTV",
        "attr:StreamingMovies",
        "shap:StreamingMovies",
        "prediction:risk",
    },
    "BILLING_REVIEW": {
        "attr:MonthlyCharges",
        "shap:MonthlyCharges",
        "attr:TotalCharges",
        "shap:TotalCharges",
        "attr:PaymentMethod",
        "shap:PaymentMethod",
        "attr:PaperlessBilling",
        "shap:PaperlessBilling",
        "prediction:risk",
    },
    "PLAN_VALUE_REVIEW": {
        "attr:InternetService",
        "shap:InternetService",
        "attr:PhoneService",
        "shap:PhoneService",
        "attr:MonthlyCharges",
        "shap:MonthlyCharges",
        "prediction:risk",
    },
    "SECURITY_SERVICE_REVIEW": {
        "attr:OnlineSecurity",
        "shap:OnlineSecurity",
        "attr:DeviceProtection",
        "shap:DeviceProtection",
        "prediction:risk",
    },
    "NO_ACTION": {"prediction:risk"},
    "HUMAN_REVIEW": {"prediction:risk"},
}

# Maps reason codes to the action IDs they may justify.
REASON_ACTION_COMPAT: dict[str, set[str]] = {
    ReasonCode.EARLY_TENURE: {"ONBOARDING_SUPPORT", "HUMAN_REVIEW"},
    ReasonCode.MONTH_TO_MONTH: {"CONTRACT_REVIEW", "HUMAN_REVIEW"},
    ReasonCode.NO_TECH_SUPPORT: {"TECH_SUPPORT_REVIEW", "HUMAN_REVIEW"},
    ReasonCode.NO_ONLINE_SECURITY: {"SECURITY_SERVICE_REVIEW", "HUMAN_REVIEW"},
    ReasonCode.HIGH_MONTHLY_CHARGES: {
        "BILLING_REVIEW",
        "PLAN_VALUE_REVIEW",
        "HUMAN_REVIEW",
    },
    ReasonCode.SERVICE_CUSTOMER: {
        "SERVICE_QUALITY_REVIEW",
        "PLAN_VALUE_REVIEW",
        "HUMAN_REVIEW",
    },
    ReasonCode.SHAP_RISK_FACTOR: {
        "ONBOARDING_SUPPORT",
        "CONTRACT_REVIEW",
        "TECH_SUPPORT_REVIEW",
        "SERVICE_QUALITY_REVIEW",
        "BILLING_REVIEW",
        "PLAN_VALUE_REVIEW",
        "SECURITY_SERVICE_REVIEW",
        "HUMAN_REVIEW",
    },
    ReasonCode.PREDICTED_CHURN_RISK: {
        "ONBOARDING_SUPPORT",
        "CONTRACT_REVIEW",
        "TECH_SUPPORT_REVIEW",
        "SERVICE_QUALITY_REVIEW",
        "BILLING_REVIEW",
        "PLAN_VALUE_REVIEW",
        "SECURITY_SERVICE_REVIEW",
        "HUMAN_REVIEW",
    },
    ReasonCode.INSUFFICIENT_EVIDENCE: {"HUMAN_REVIEW"},
    ReasonCode.RETAINED_CLASS: {"NO_ACTION"},
}

DIRECT_ACTIONS = {
    "ONBOARDING_SUPPORT": ("attr:tenure", ReasonCode.EARLY_TENURE),
    "CONTRACT_REVIEW": ("attr:Contract", ReasonCode.MONTH_TO_MONTH),
    "TECH_SUPPORT_REVIEW": ("attr:TechSupport", ReasonCode.NO_TECH_SUPPORT),
    "SECURITY_SERVICE_REVIEW": (
        "attr:OnlineSecurity",
        ReasonCode.NO_ONLINE_SECURITY,
    ),
}
SHAP_ACTION_FEATURES = {
    "BILLING_REVIEW": {"MonthlyCharges", "TotalCharges", "PaymentMethod"},
    "PLAN_VALUE_REVIEW": {"MonthlyCharges", "InternetService"},
    "SERVICE_QUALITY_REVIEW": {"InternetService"},
}


def available_reason_codes(context: dict) -> list[str]:
    """Expose only reason codes supported by this minimized context."""
    if context["predicted_churn"] == 0:
        return [ReasonCode.RETAINED_CLASS.value]
    attributes = context["attributes"]
    codes = [
        ReasonCode.PREDICTED_CHURN_RISK.value,
        ReasonCode.INSUFFICIENT_EVIDENCE.value,
    ]
    predicates = {
        ReasonCode.EARLY_TENURE: int(attributes.get("tenure", 999)) <= 6,
        ReasonCode.MONTH_TO_MONTH: attributes.get("Contract") == "Month-to-month",
        ReasonCode.NO_TECH_SUPPORT: attributes.get("TechSupport") == "No",
        ReasonCode.NO_ONLINE_SECURITY: attributes.get("OnlineSecurity") == "No",
    }
    codes.extend(reason.value for reason, matches in predicates.items() if matches)
    if any(f.get("direction") == "increasing" for f in context["factors"]):
        codes.append(ReasonCode.SHAP_RISK_FACTOR.value)
    return codes


def validate_llm_selection(
    selection: LLMSelection | dict, context: dict
) -> LLMSelection:
    """Validate an LLM selection against context.

    Checks action eligibility, evidence existence, reason-code compatibility,
    action-evidence compatibility, and demographic safeguards.
    """
    item = (
        selection
        if isinstance(selection, LLMSelection)
        else LLMSelection.model_validate(selection)
    )
    _check_prompt_injection(context)
    if len(set(item.evidence_ids)) != len(item.evidence_ids) or len(
        set(item.reason_codes)
    ) != len(item.reason_codes):
        raise ValueError("Duplicate evidence or reason code")

    # 1. Action eligibility
    action = item.action_id.value
    if action not in context.get("eligible_actions", {}):
        raise ValueError(f"Ineligible action: {action}")

    # 2. Evidence existence in context
    allowed_refs = (
        {f"attr:{name}" for name in context.get("attributes", {})}
        | {f["ref"] for f in context.get("factors", [])}
        | {"prediction:risk"}
    )
    for ref in item.evidence_ids:
        if ref not in allowed_refs:
            raise ValueError(f"Unsupported evidence reference: {ref}")

    # 3. No demographic evidence
    for ref in item.evidence_ids:
        for demo in DEMOGRAPHIC_ATTRIBUTES:
            if ref == f"attr:{demo}" or ref == f"shap:{demo}":
                raise ValueError(f"Demographic attribute in evidence: {ref}")

    # 4. Action-evidence compatibility
    compatible_evidence = ACTION_EVIDENCE_RULES.get(action)
    if compatible_evidence is not None and action not in ("HUMAN_REVIEW", "NO_ACTION"):
        if not any(ref in compatible_evidence for ref in item.evidence_ids):
            raise ValueError(
                f"No evidence in {item.evidence_ids} supports action {action}"
            )

    attributes = context.get("attributes", {})
    reasons = set(item.reason_codes)
    evidence = set(item.evidence_ids)
    direct_truth = {
        ReasonCode.EARLY_TENURE: int(attributes.get("tenure", 999)) <= 6,
        ReasonCode.MONTH_TO_MONTH: attributes.get("Contract") == "Month-to-month",
        ReasonCode.NO_TECH_SUPPORT: attributes.get("TechSupport") == "No",
        ReasonCode.NO_ONLINE_SECURITY: attributes.get("OnlineSecurity") == "No",
    }
    for selected_action, (anchor, reason) in DIRECT_ACTIONS.items():
        if action == selected_action and (
            anchor not in evidence or reason not in reasons or not direct_truth[reason]
        ):
            raise ValueError(f"Action {action} lacks its observed evidence and reason")
    risk_factors = {
        f["ref"]: f["feature"]
        for f in context.get("factors", [])
        if f.get("direction") == "increasing"
    }
    if action in SHAP_ACTION_FEATURES and (
        ReasonCode.SHAP_RISK_FACTOR not in reasons
        or not any(
            risk_factors.get(ref) in SHAP_ACTION_FEATURES[action] for ref in evidence
        )
    ):
        raise ValueError(
            f"Action {action} needs a matching risk-increasing SHAP factor"
        )
    if action == "HUMAN_REVIEW" and (
        ReasonCode.INSUFFICIENT_EVIDENCE not in reasons
        or "prediction:risk" not in evidence
    ):
        raise ValueError("HUMAN_REVIEW requires risk evidence and review reason")
    if action == "NO_ACTION" and (
        ReasonCode.RETAINED_CLASS not in reasons or "prediction:risk" not in evidence
    ):
        raise ValueError("NO_ACTION requires retained-class evidence and reason")

    # 5. Reason-code compatibility
    for rc in item.reason_codes:
        if rc == ReasonCode.HIGH_MONTHLY_CHARGES:
            raise ValueError("High charges has no approved threshold")
        if rc.value not in available_reason_codes(context):
            raise ValueError(f"Reason code {rc.value} has no contextual support")
        compatible_actions = REASON_ACTION_COMPAT.get(rc.value, set())
        if action not in compatible_actions:
            raise ValueError(
                f"Reason code {rc.value} is incompatible with action {action}"
            )
        if rc in direct_truth and (
            not direct_truth[rc]
            or DIRECT_ACTIONS[
                next(a for a, (_, reason) in DIRECT_ACTIONS.items() if reason == rc)
            ][0]
            not in evidence
        ):
            raise ValueError(f"Reason code {rc.value} contradicts observed evidence")
        if rc == ReasonCode.SHAP_RISK_FACTOR and not any(
            ref in risk_factors for ref in evidence
        ):
            raise ValueError("SHAP reason lacks selected risk-increasing factor")
        if rc == ReasonCode.PREDICTED_CHURN_RISK and (
            context["predicted_churn"] != 1 or "prediction:risk" not in evidence
        ):
            raise ValueError("Churn-risk reason lacks prediction evidence")
        if rc == ReasonCode.SERVICE_CUSTOMER and (
            attributes.get("InternetService") in (None, "No")
            or "attr:InternetService" not in evidence
        ):
            raise ValueError("Service-customer reason lacks observed service")

    # 6. HUMAN_REVIEW must have human_review_required=True
    if (
        item.human_review_required
        != context["eligible_actions"][action]["approval_required"]
    ):
        raise ValueError("Human review flag violates policy")

    return item


def validate_recommendation(
    value: Recommendation | dict, context: dict
) -> Recommendation:
    """Validate a fully rendered recommendation against context and policy."""
    item = Recommendation.model_validate(value)
    rule = context["eligible_actions"].get(item.action_id.value)
    if rule is None:
        raise ValueError("Ineligible action")
    if (
        item.priority.value != rule["priority"]
        or item.channel.value not in rule["channels"]
    ):
        raise ValueError("Action priority or channel violates policy")
    if item.human_approval_required != rule["approval_required"]:
        raise ValueError("Approval requirement violates policy")
    allowed_refs = (
        {f"attr:{name}" for name in context["attributes"]}
        | {f["ref"] for f in context["factors"]}
        | {"prediction:risk"}
    )
    if any(ref not in allowed_refs for ref in item.evidence_refs):
        raise ValueError("Unsupported evidence reference")
    if item.reason_codes:
        validate_llm_selection(
            {
                "action_id": item.action_id,
                "evidence_ids": item.evidence_refs,
                "reason_codes": item.reason_codes,
                "human_review_required": item.human_approval_required,
            },
            context,
        )

    # Demographic safeguard for evidence
    for ref in item.evidence_refs:
        for demo in DEMOGRAPHIC_ATTRIBUTES:
            if ref == f"attr:{demo}" or ref == f"shap:{demo}":
                raise ValueError(f"Demographic attribute in evidence: {ref}")

    text = " ".join(
        [item.summary, item.rationale, item.limitations, *item.talking_points]
    )
    if FORBIDDEN.search(text) or GUARANTEE.search(NEGATED_GUARANTEE.sub("", text)):
        raise ValueError("Unsafe or unsupported claim")
    if item.action_id.value == "NO_ACTION" and item.channel.value != "none":
        raise ValueError("No-action must have no communication channel")

    # Prompt injection: reject context values that look like instructions
    _check_prompt_injection(context)

    return item


def _check_prompt_injection(context: dict) -> None:
    """Reject context that contains potential injection patterns."""
    injection_patterns = re.compile(
        r"(ignore\s+(all\s+)?previous|override\s+policy|system:\s*|"
        r"you\s+are\s+now|forget\s+your\s+instructions|"
        r"disregard\s+all|new\s+instructions:)",
        re.I,
    )
    for key, value in context.get("attributes", {}).items():
        if isinstance(value, str) and injection_patterns.search(value):
            raise ValueError(f"Potential prompt injection in attribute: {key}")
