"""Offline Phase 9B regression on the preserved 24 cases and four holdouts."""

import json
from pathlib import Path
from types import SimpleNamespace

import mlflow

from retention.genai.config import load_genai_config, load_policies
from retention.genai.evaluation import evaluate_cases
from retention.genai.guardrails import validate_llm_selection, validate_recommendation
from retention.genai.renderer import RENDERING_VERSION, render_recommendation
from retention.genai.schemas import LLMSelection
from retention.genai.service import _rule_recommendation
from scripts.verify_genai_live import HOLDOUT, SOURCE, synthetic_context

OUTPUT = Path("reports/generated/genai/phase9b_offline_v3.json")


def mock_selection(context):
    """Exercise the contract with observed predicates, not expected labels."""
    attributes = context["attributes"]
    if context["predicted_churn"] == 0:
        action, ref, reason, human = (
            "NO_ACTION",
            "prediction:risk",
            "RETAINED_CLASS",
            False,
        )
    elif int(attributes["tenure"]) <= 6:
        action, ref, reason, human = (
            "ONBOARDING_SUPPORT",
            "attr:tenure",
            "EARLY_TENURE",
            True,
        )
    elif attributes["Contract"] == "Month-to-month":
        action, ref, reason, human = (
            "CONTRACT_REVIEW",
            "attr:Contract",
            "MONTH_TO_MONTH",
            True,
        )
    elif attributes["TechSupport"] == "No":
        action, ref, reason, human = (
            "TECH_SUPPORT_REVIEW",
            "attr:TechSupport",
            "NO_TECH_SUPPORT",
            True,
        )
    elif attributes["OnlineSecurity"] == "No":
        action, ref, reason, human = (
            "SECURITY_SERVICE_REVIEW",
            "attr:OnlineSecurity",
            "NO_ONLINE_SECURITY",
            True,
        )
    else:
        action, ref, reason, human = (
            "HUMAN_REVIEW",
            "prediction:risk",
            "INSUFFICIENT_EVIDENCE",
            True,
        )
    return validate_llm_selection(
        LLMSelection(
            action_id=action,
            evidence_ids=[ref],
            reason_codes=[reason],
            human_review_required=human,
        ),
        context,
    )


def evaluate(rows, policies):
    cases = [{**row, "context": synthetic_context(row, policies)} for row in rows]

    def rules(case):
        return SimpleNamespace(
            recommendation=_rule_recommendation(case["context"], fallback=False),
            generation_source="rules",
            latency_ms=0.0,
        )

    def mocked(case):
        context = case["context"]
        if case["expected_source"] == "fallback":
            item = _rule_recommendation(context, fallback=True)
            source = "simulated_fallback"
        else:
            selection = mock_selection(context)
            item = validate_recommendation(
                render_recommendation(
                    action_id=selection.action_id.value,
                    evidence_ids=selection.evidence_ids,
                    reason_codes=[reason.value for reason in selection.reason_codes],
                    human_review=selection.human_review_required,
                    context=context,
                    policy_rule=context["eligible_actions"][selection.action_id.value],
                ),
                context,
            )
            source = "mocked_selection"
        return SimpleNamespace(
            recommendation=item, generation_source=source, latency_ms=0.0
        )

    return {
        "conservative_rules": evaluate_cases(cases, rules),
        "mocked_contract": evaluate_cases(cases, mocked),
    }


def main():
    config = load_genai_config()
    policies = load_policies()
    result = {
        "evaluation_type": "offline_regression_only",
        "provider_calls": 0,
        "provider": config.provider,
        "configured_model": config.model,
        "prompt_version": config.prompt_version,
        "policy_version": config.policy_version,
        "rendering_version": RENDERING_VERSION,
        "original_24": evaluate(json.loads(SOURCE.read_text()), policies),
        "holdout_4": evaluate(json.loads(HOLDOUT.read_text()), policies),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("customer-retention-genai-nba")
    with mlflow.start_run(run_name="phase9b-offline-v3") as run:
        mlflow.log_params(
            {
                key: result[key]
                for key in (
                    "evaluation_type",
                    "provider_calls",
                    "provider",
                    "configured_model",
                    "prompt_version",
                    "policy_version",
                    "rendering_version",
                )
            }
        )
        for dataset in ("original_24", "holdout_4"):
            for source, metrics in result[dataset].items():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{dataset}.{source}.{key}", value)
        mlflow.log_artifact(str(OUTPUT), artifact_path="genai")
        result["mlflow_run_id"] = run.info.run_id
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
