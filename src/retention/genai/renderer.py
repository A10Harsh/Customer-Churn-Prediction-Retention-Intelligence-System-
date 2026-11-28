"""Deterministic recommendation text from validated evidence and templates.

All recommendation text is generated from verified customer attributes,
prediction results, SHAP contributions, and business policies.  The LLM
selects which action and evidence to use; this module renders the final
human-readable recommendation without any LLM-generated prose.
"""

from __future__ import annotations

from retention.genai.schemas import ActionID

RENDERING_VERSION = "1"

# ---------------------------------------------------------------------------
# Action templates
# ---------------------------------------------------------------------------

_SUMMARIES: dict[str, str] = {
    ActionID.NO_ACTION: "No proactive retention action indicated.",
    ActionID.ONBOARDING_SUPPORT: (
        "Consider onboarding support based on observed early tenure."
    ),
    ActionID.CONTRACT_REVIEW: (
        "Consider contract review based on observed month-to-month agreement."
    ),
    ActionID.TECH_SUPPORT_REVIEW: (
        "Consider technical support review based on observed service attributes."
    ),
    ActionID.SERVICE_QUALITY_REVIEW: (
        "Consider service quality review based on observed internet service."
    ),
    ActionID.BILLING_REVIEW: (
        "Consider billing review based on predictive billing attribution."
    ),
    ActionID.PLAN_VALUE_REVIEW: (
        "Consider plan value review based on observed service attributes."
    ),
    ActionID.SECURITY_SERVICE_REVIEW: (
        "Consider online security review based on observed service attributes."
    ),
    ActionID.HUMAN_REVIEW: (
        "Route to human review: insufficient automated evidence for action selection."
    ),
}

_RATIONALE_TEMPLATES: dict[str, str] = {
    ActionID.NO_ACTION: "The saved calibrated prediction class is retained.",
    ActionID.ONBOARDING_SUPPORT: (
        "The customer's observed tenure is {tenure} months. "
        "The prediction model classified this customer as at-risk."
    ),
    ActionID.CONTRACT_REVIEW: (
        "The customer's observed contract type is {Contract}. "
        "The prediction model classified this customer as at-risk."
    ),
    ActionID.TECH_SUPPORT_REVIEW: (
        "The customer's observed TechSupport attribute is {TechSupport}. "
        "The prediction model classified this customer as at-risk."
    ),
    ActionID.SERVICE_QUALITY_REVIEW: (
        "The customer's observed internet service is {InternetService}. "
        "The prediction model classified this customer as at-risk."
    ),
    ActionID.BILLING_REVIEW: (
        "The customer's observed monthly charges are {MonthlyCharges}. "
        "The prediction model classified this customer as at-risk."
    ),
    ActionID.PLAN_VALUE_REVIEW: (
        "The customer's observed internet service is {InternetService}. "
        "The prediction model classified this customer as at-risk."
    ),
    ActionID.SECURITY_SERVICE_REVIEW: (
        "The customer's observed OnlineSecurity attribute is {OnlineSecurity}. "
        "The prediction model classified this customer as at-risk."
    ),
    ActionID.HUMAN_REVIEW: (
        "The available evidence is insufficient for automated action selection. "
        "A representative should review the case."
    ),
}

_TALKING_POINTS: dict[str, list[str]] = {
    ActionID.NO_ACTION: [],
    ActionID.ONBOARDING_SUPPORT: [
        "Review the customer's onboarding experience based on observed tenure.",
        "Confirm whether the customer has questions about their current services.",
    ],
    ActionID.CONTRACT_REVIEW: [
        "Review the customer's contract options based on observed agreement type.",
        "Confirm whether contract flexibility is relevant to the customer.",
    ],
    ActionID.TECH_SUPPORT_REVIEW: [
        "Ask whether the customer has technical support needs.",
        "Review the customer's current support-service status.",
    ],
    ActionID.SERVICE_QUALITY_REVIEW: [
        "Review the customer's service experience based on observed service type.",
        "Confirm whether service-related concerns have been raised.",
    ],
    ActionID.BILLING_REVIEW: [
        "Review the customer's billing based on observed charges.",
        "Confirm whether billing concerns have been raised.",
    ],
    ActionID.PLAN_VALUE_REVIEW: [
        "Ask whether the current plan meets the customer's needs.",
        "Review the current service configuration with the customer.",
    ],
    ActionID.SECURITY_SERVICE_REVIEW: [
        "Ask whether the customer has security-service questions.",
        "Review the current online-security status with the customer.",
    ],
    ActionID.HUMAN_REVIEW: [
        "A representative should review all available evidence for this case.",
    ],
}

_CHANNEL: dict[str, str] = {
    ActionID.NO_ACTION: "none",
    ActionID.HUMAN_REVIEW: "internal",
}

_LIMITATIONS = (
    "This recommendation is based on predictive modelling, not causal analysis. "
    "SHAP contributions describe raw-margin attribution, not probability changes. "
    "Human approval is required before any intervention."
)

_NO_ACTION_LIMITATIONS = (
    "Predictive risk is not a causal finding or guaranteed outcome."
)

_HUMAN_REVIEW_LIMITATIONS = (
    "Automated evidence was insufficient for action selection. "
    "Predictive risk is not a causal finding or guaranteed outcome."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_recommendation(
    action_id: str,
    evidence_ids: list[str],
    reason_codes: list[str],
    human_review: bool,
    context: dict,
    policy_rule: dict,
) -> dict:
    """Render a full recommendation dict from validated selection fields.

    Returns a dict suitable for ``Recommendation.model_validate()``.
    All text is deterministic and grounded in ``context``.
    """
    from retention.genai.guardrails import (
        SHAP_ACTION_FEATURES,
        validate_llm_selection,
    )

    selection = validate_llm_selection(
        {
            "action_id": action_id,
            "evidence_ids": evidence_ids,
            "reason_codes": reason_codes,
            "human_review_required": human_review,
        },
        context,
    )
    action_id = selection.action_id.value
    if policy_rule != context["eligible_actions"][action_id]:
        raise ValueError("Rendering policy rule does not match context")
    rendered_refs = list(evidence_ids)
    if "prediction:risk" not in rendered_refs:
        if len(rendered_refs) >= 6:
            raise ValueError("No space for required prediction evidence")
        rendered_refs.append("prediction:risk")
    attributes = context.get("attributes", {})

    # --- Summary ---
    summary = _SUMMARIES.get(action_id, _SUMMARIES[ActionID.HUMAN_REVIEW])

    # --- Rationale (template + attributes only) ---
    template = _RATIONALE_TEMPLATES.get(
        action_id, _RATIONALE_TEMPLATES[ActionID.HUMAN_REVIEW]
    )
    # Only supply known, safe attribute values
    safe_values = {k: str(v) for k, v in attributes.items()}
    try:
        rationale = template.format_map(safe_values)
    except KeyError as exc:
        raise ValueError("Required rendering attribute is missing") from exc
    if action_id in SHAP_ACTION_FEATURES:
        factor = next(
            f
            for f in context["factors"]
            if f["ref"] in evidence_ids
            and f["direction"] == "increasing"
            and f["feature"] in SHAP_ACTION_FEATURES[action_id]
        )
        rationale = (
            f"The supplied SHAP attribution for {factor['feature']} increased "
            "the model's raw risk margin. This is predictive, not causal."
        )

    # --- Talking points ---
    talking_points = _TALKING_POINTS.get(
        action_id, _TALKING_POINTS[ActionID.HUMAN_REVIEW]
    )

    # --- Channel: use policy if available, else default ---
    channels = policy_rule.get("channels", ["internal"])
    channel = _CHANNEL.get(action_id, channels[0] if channels else "internal")

    # --- Priority and approval from policy ---
    priority = policy_rule.get("priority", "high")
    approval = policy_rule.get("approval_required", True)
    if human_review:
        approval = True

    # --- Limitations ---
    if action_id == ActionID.NO_ACTION:
        limitations = _NO_ACTION_LIMITATIONS
    elif action_id == ActionID.HUMAN_REVIEW:
        limitations = _HUMAN_REVIEW_LIMITATIONS
    else:
        limitations = _LIMITATIONS

    return {
        "action_id": action_id,
        "priority": priority,
        "channel": channel,
        "summary": summary,
        "rationale": rationale,
        "evidence_refs": rendered_refs,
        "talking_points": talking_points,
        "human_approval_required": approval,
        "limitations": limitations,
        "reason_codes": [str(rc) for rc in reason_codes],
        "rendering_version": RENDERING_VERSION,
    }
