"""Deterministic eligibility over actionable, non-demographic attributes."""

from retention.genai.schemas import ActionID


def eligible_actions(attributes: dict, predicted_churn: int, policies: dict) -> dict:
    rules = policies["actions"]
    if predicted_churn == 0:
        return {ActionID.NO_ACTION.value: rules[ActionID.NO_ACTION.value]}
    if predicted_churn != 1:
        raise ValueError("Invalid predicted class")
    tests = {
        "always": True,
        "all_churn": True,
        "early_tenure": int(attributes["tenure"]) <= 6,
        "month_to_month": attributes["Contract"] == "Month-to-month",
        "no_tech_support": attributes["TechSupport"] == "No",
        "service_customer": attributes["InternetService"] != "No",
        "no_online_security": attributes["OnlineSecurity"] == "No",
        "retained": False,
    }
    return {name: rule for name, rule in rules.items() if tests[rule["eligibility"]]}
