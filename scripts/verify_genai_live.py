"""Sanitized, synthetic-only Gemini verification."""

import argparse
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from retention.genai.config import load_genai_config, load_policies
from retention.genai.guardrails import (
    validate_llm_selection,
    validate_recommendation,
)
from retention.genai.llm import make_llm
from retention.genai.policies import eligible_actions
from retention.genai.prompts import prompt_for
from retention.genai.renderer import RENDERING_VERSION, render_recommendation
from retention.genai.schemas import LLMSelection, Recommendation
from retention.genai.service import _rule_recommendation

SOURCE = Path("reports/generated/genai/synthetic_scenarios_v1.json")
HOLDOUT = Path("reports/generated/genai/prompt_holdout_v1.json")


def synthetic_context(row, policies):
    """Reproduce the v1 offline scenario construction exactly."""
    attributes = row.get("attributes") or {
        "tenure": 3 if row["scenario"] == "onboarding" else 24,
        "Contract": "Month-to-month" if row["scenario"] == "contract" else "One year",
        "TechSupport": "No" if row["scenario"] == "tech_support" else "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "No" if row["scenario"] == "security" else "Yes",
    }
    return {
        "predicted_churn": row["predicted_churn"],
        "attributes": attributes,
        "factors": [],
        "eligible_actions": eligible_actions(
            attributes, row["predicted_churn"], policies
        ),
    }


def failure_category(exc):
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if any(word in name + message for word in ("quota", "resourceexhausted", "429")):
        return "quota"
    if any(
        word in name + message
        for word in ("unauthorized", "forbidden", "401", "403", "api key")
    ):
        return "authentication"
    if any(word in name + message for word in ("notfound", "404", "unsupported model")):
        return "unsupported_model"
    if any(word in name + message for word in ("503", "unavailable")):
        return "provider_unavailable"
    if any(word in name + message for word in ("timeout", "deadline")):
        return "timeout"
    return "provider_or_parsing"


def percentile(values, percentage):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = int(position)
    return ordered[lower] + (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    ) * (position - lower)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    config = load_genai_config()
    policies = load_policies()
    rows = json.loads((HOLDOUT if args.holdout else SOURCE).read_text())
    output = Path(
        "reports/generated/genai/"
        + ("prompt_holdout_evaluation_v" if args.holdout else "live_evaluation_v")
        + config.prompt_version
        + ".json"
    )
    # Phase 9B has at most two calls here, with SDK retries disabled. A third
    # provider attempt remains unspent under the user-specified global budget.
    live_config = config.model_copy(update={"max_retries": 0})
    client = make_llm(live_config)
    schema = LLMSelection if config.prompt_version == "3" else Recommendation
    structured = client.with_structured_output(schema, include_raw=True)
    if config.prompt_version == "3":
        selected = rows[:2] if args.holdout else [rows[0], rows[1], rows[2], rows[11]]
        if args.smoke:
            selected = selected[1:2]
    else:
        selected = [rows[2]] if args.smoke else rows
    records = []
    for row in selected:
        context = synthetic_context(row, policies)
        entry = {"id": row["id"], "scenario": row["scenario"]}
        if row["predicted_churn"] == 0:
            item = _rule_recommendation(context, fallback=False)
            entry.update(
                source="rules", action=item.action_id.value, policy_compliant=True
            )
        elif row["expected_source"] == "fallback":
            item = _rule_recommendation(context, fallback=True)
            entry.update(
                source="simulated_fallback",
                action=item.action_id.value,
                policy_compliant=True,
            )
        else:
            start = time.perf_counter()
            try:
                response = structured.invoke(prompt_for(context, config.prompt_version))
                entry["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
                parsed = response.get("parsed")
                if parsed is None:
                    raise ValueError("No structured result")
                item = schema.model_validate(parsed)
                entry["schema_valid"] = True
                entry["action"] = item.action_id.value
                if config.prompt_version == "3":
                    entry["selection"] = item.model_dump(mode="json")
                else:
                    entry["recommendation"] = item.model_dump(mode="json")
                raw = response.get("raw")
                entry["actual_model"] = getattr(raw, "response_metadata", {}).get(
                    "model_name"
                )
                usage = getattr(raw, "usage_metadata", None) or {}
                entry["input_tokens"] = usage.get("input_tokens")
                entry["output_tokens"] = usage.get("output_tokens")
                try:
                    if config.prompt_version == "3":
                        selection = validate_llm_selection(item, context)
                        rendered = render_recommendation(
                            action_id=selection.action_id.value,
                            evidence_ids=selection.evidence_ids,
                            reason_codes=[x.value for x in selection.reason_codes],
                            human_review=selection.human_review_required,
                            context=context,
                            policy_rule=context["eligible_actions"][
                                selection.action_id.value
                            ],
                        )
                        item = Recommendation.model_validate(rendered)
                        entry["recommendation"] = item.model_dump(mode="json")
                    validate_recommendation(item, context)
                    entry["policy_compliant"] = True
                except (ValueError, TypeError) as exc:
                    entry["policy_compliant"] = False
                    entry["validation_failure"] = str(exc)
                entry["source"] = "llm"
            except Exception as exc:
                entry.update(source="failed", error_category=failure_category(exc))
                entry["error_type"] = type(exc).__name__
                entry["error_detail"] = str(exc).replace(
                    os.getenv("GOOGLE_API_KEY") or "__never_match__", "[REDACTED]"
                )[:300]
                entry["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        if "action" in entry:
            entry["preferred_agreement"] = (
                entry["action"] == row["acceptable_actions"][0]
            )
            entry["acceptable_agreement"] = entry["action"] in row["acceptable_actions"]
            entry["eligible"] = entry["action"] in context["eligible_actions"]
        records.append(entry)
        print(
            json.dumps({k: v for k, v in entry.items() if k != "recommendation"}),
            flush=True,
        )
        if entry.get("error_category") in {
            "quota",
            "authentication",
            "unsupported_model",
        }:
            break
    llm = [r for r in records if r["source"] == "llm"]
    attempted = [r for r in records if r["source"] in {"llm", "failed"}]
    latencies = [r["latency_ms"] for r in llm]
    counts = Counter(r["source"] for r in records)
    metrics = {
        "scenarios_total": len(rows),
        "scenarios_processed": len(records),
        "llm_applicable": sum(
            r["predicted_churn"] == 1 and r["expected_source"] != "fallback"
            for r in rows
        ),
        "llm_attempted": len(attempted),
        "llm_generated": len(llm),
        "llm_accepted": sum(r.get("policy_compliant", False) for r in llm),
        "policy_rejected": sum(not r.get("policy_compliant", False) for r in llm),
        "llm_failed": counts["failed"],
        "deterministic_no_action": counts["rules"],
        "simulated_fallback": counts["simulated_fallback"],
        "schema_valid": sum(r.get("schema_valid", False) for r in llm),
        "policy_compliant": sum(r.get("policy_compliant", False) for r in llm),
        "eligible": sum(r.get("eligible", False) for r in llm),
        "preferred_agreement": sum(r.get("preferred_agreement", False) for r in llm),
        "acceptable_agreement": sum(r.get("acceptable_agreement", False) for r in llm),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95),
        "input_tokens_reported": sum(r.get("input_tokens") is not None for r in llm),
        "output_tokens_reported": sum(r.get("output_tokens") is not None for r in llm),
        "input_tokens_total": sum(r.get("input_tokens") or 0 for r in llm),
        "output_tokens_total": sum(r.get("output_tokens") or 0 for r in llm),
    }
    result = {
        "evaluation_type": (
            "live_gemini_smoke"
            if args.smoke
            else "phase9b_limited_live"
            if config.prompt_version == "3"
            else "refined_prompt_holdout"
            if args.holdout
            else "live_gemini_v1"
        ),
        "provider": config.provider,
        "configured_model": config.model,
        "prompt_version": config.prompt_version,
        "policy_version": config.policy_version,
        "rendering_version": RENDERING_VERSION
        if config.prompt_version == "3"
        else "legacy",
        "scenario_version": "holdout-1" if args.holdout else "1",
        "metrics": metrics,
        "failure_categories": dict(
            Counter(r["error_category"] for r in records if "error_category" in r)
        ),
        "records": records,
    }
    if not args.smoke or config.prompt_version == "3":
        output.write_text(json.dumps(result, indent=2) + "\n")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("customer-retention-genai-nba")
        with mlflow.start_run(
            run_name=f"phase9-live-gemini-prompt-v{config.prompt_version}"
        ) as run:
            mlflow.log_params(
                {
                    k: result[k]
                    for k in (
                        "evaluation_type",
                        "provider",
                        "configured_model",
                        "prompt_version",
                        "policy_version",
                        "rendering_version",
                        "scenario_version",
                    )
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            mlflow.log_dict(
                result["failure_categories"], "genai/failure_categories.json"
            )
            mlflow.log_artifact(str(output), artifact_path="genai")
            result["mlflow_run_id"] = run.info.run_id
        output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
