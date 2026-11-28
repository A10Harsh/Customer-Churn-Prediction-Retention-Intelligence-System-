"""Exercise synthetic prediction, explanation, and recommendation persistence."""

import json
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import func, select, text

from retention.database.connection import (
    create_database_engine,
    make_session_factory,
    session_scope,
)
from retention.database.explanations import explain_prediction
from retention.database.inference import predict_and_persist
from retention.database.models import (
    Customer,
    Explanation,
    Prediction,
    RecommendationRecord,
)
from retention.genai.service import recommend_prediction, recommendation_history
from tests.integration.test_database import synthetic_customer

OUTPUT = Path("reports/generated/genai/e2e_synthetic_v2.json")


class FailingModel:
    def with_structured_output(self, schema, include_raw=False):
        return self

    def invoke(self, prompt):
        raise RuntimeError("Simulated provider failure")


def counts(session):
    return {
        table.__tablename__: session.scalar(select(func.count()).select_from(table))
        for table in (Customer, Prediction, Explanation, RecommendationRecord)
    }


def main():
    load_dotenv()
    engine = create_database_engine()
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        version = session.scalar(text("SELECT version_num FROM alembic_version"))
        if version != "0003_recommendations":
            raise RuntimeError("Application database migration 0003 is required")
        before = counts(session)
    base = synthetic_customer()
    high = dict(
        base,
        tenure=1,
        Contract="Month-to-month",
        TechSupport="No",
        OnlineSecurity="No",
        MonthlyCharges="95.00",
        TotalCharges="95.00",
    )
    retained = dict(
        base,
        tenure=60,
        Contract="Two year",
        TechSupport="Yes",
        OnlineSecurity="Yes",
        MonthlyCharges="45.00",
        TotalCharges="2700.00",
    )
    report = {"database_version": version, "before": before, "cases": {}}
    for name, values in (
        ("live_high", high),
        ("retained", retained),
        ("failure", high),
    ):
        external_id = "PHASE9-VERIFY-" + uuid4().hex
        prediction_id, prediction = predict_and_persist(factory, external_id, values)
        explanation = explain_prediction(factory, prediction_id)
        if name == "failure":
            result = recommend_prediction(factory, prediction_id, llm=FailingModel())
        else:
            result = recommend_prediction(factory, prediction_id)
        again = recommend_prediction(factory, prediction_id, llm=FailingModel())
        entry = {
            "prediction_id": prediction_id,
            "explanation_id": explanation.explanation_id,
            "predicted_churn": prediction["predicted_churn"],
            "probability": prediction["churn_probability"],
            "recommendation_id": result.id,
            "source": result.generation_source,
            "action": result.recommendation.action_id.value,
            "evidence_refs": result.recommendation.evidence_refs,
            "provider": result.provider,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "idempotent": result.id == again.id,
        }
        if name == "failure":
            regenerated = recommend_prediction(
                factory, prediction_id, llm=FailingModel(), regenerate_fallback=True
            )
            entry["regenerated_id"] = regenerated.id
            entry["regeneration_new_attempt"] = regenerated.id != result.id
        with session_scope(factory) as session:
            entry["history_count"] = len(recommendation_history(session, prediction_id))
        report["cases"][name] = entry
        print(json.dumps({name: entry}), flush=True)
    with session_scope(factory) as session:
        report["after"] = counts(session)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    engine.dispose()


if __name__ == "__main__":
    main()
