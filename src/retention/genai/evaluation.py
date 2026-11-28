"""Deterministic evaluation of policy and grounding on synthetic scenarios.

Separates actual LLM results from mocked and deterministic results.
Never counts fallback as successful LLM generation.
"""

from collections import Counter

from retention.genai.guardrails import validate_recommendation


def evaluate_cases(cases: list[dict], recommend) -> dict:
    results = Counter()
    latencies = []
    for case in cases:
        results["total"] += 1
        try:
            output = recommend(case)
            item = validate_recommendation(output.recommendation, case["context"])
            results["valid"] += 1
            if item.action_id.value in case["acceptable_actions"]:
                results["expert_agreement"] += 1
            if output.generation_source == "fallback":
                results["fallback"] += 1
            elif output.generation_source == "llm":
                results["llm_generated"] += 1
            elif output.generation_source == "rules":
                results["deterministic"] += 1
            if case.get("expected_source") is not None:
                results["expected_source_cases"] += 1
                if output.generation_source == case["expected_source"]:
                    results["expected_source_match"] += 1
            latencies.append(output.latency_ms)
        except Exception:
            results["failures"] += 1
    return {
        **dict(results),
        "valid_rate": results["valid"] / results["total"] if results["total"] else 0,
        "agreement_rate": results["expert_agreement"] / results["total"]
        if results["total"]
        else 0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }
