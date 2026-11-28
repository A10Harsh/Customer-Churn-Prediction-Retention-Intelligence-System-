"""Offline, synthetic policy evaluation. No external API calls or customer data."""

import json
from pathlib import Path
from types import SimpleNamespace

import mlflow

from retention.genai.config import load_genai_config, load_policies
from retention.genai.evaluation import evaluate_cases
from retention.genai.policies import eligible_actions
from retention.genai.renderer import RENDERING_VERSION
from retention.genai.service import _rule_recommendation

SOURCE = Path("reports/generated/genai/synthetic_scenarios_v1.json")
OUTPUT = Path("reports/generated/genai/mock_evaluation_v3.json")


def main():
    config = load_genai_config()
    policies = load_policies()
    rows = json.loads(SOURCE.read_text())
    cases = []
    for row in rows:
        attributes = {
            "tenure": 3 if row["scenario"] == "onboarding" else 24,
            "Contract": "Month-to-month"
            if row["scenario"] == "contract"
            else "One year",
            "TechSupport": "No" if row["scenario"] == "tech_support" else "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "No" if row["scenario"] == "security" else "Yes",
        }
        context = {
            "predicted_churn": row["predicted_churn"],
            "attributes": attributes,
            "factors": [],
            "eligible_actions": eligible_actions(
                attributes, row["predicted_churn"], policies
            ),
        }
        cases.append({**row, "context": context})

    def rules(case):
        recommendation = _rule_recommendation(case["context"], fallback=False)
        return SimpleNamespace(
            recommendation=recommendation, generation_source="rules", latency_ms=0.0
        )

    metrics = evaluate_cases(cases, rules)
    result = {
        "evaluation_type": "mocked_deterministic_v3",
        "scenario_version": "1",
        "model": config.model,
        "prompt_version": config.prompt_version,
        "policy_version": config.policy_version,
        "rendering_version": RENDERING_VERSION,
        "metrics": metrics,
        "generation_source_breakdown": {
            "deterministic": metrics.get("deterministic", 0),
            "llm_generated": metrics.get("llm_generated", 0),
            "fallback": metrics.get("fallback", 0),
        },
        "live_evaluation": (
            "deferred: limited to 3 API attempts; mocked verification completed"
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("customer-retention-genai-nba")
    with mlflow.start_run(run_name="phase9b-mocked-deterministic-v3") as run:
        mlflow.log_params(
            {
                k: result[k]
                for k in (
                    "evaluation_type",
                    "scenario_version",
                    "model",
                    "prompt_version",
                    "policy_version",
                    "rendering_version",
                )
            }
        )
        mlflow.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        mlflow.log_artifact(str(OUTPUT), artifact_path="genai")
        result["mlflow_run_id"] = run.info.run_id
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
