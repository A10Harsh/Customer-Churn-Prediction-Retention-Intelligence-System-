"""Frozen, synthetic-only checks for raw-margin SHAP semantics."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from retention.explainability.aggregation import (
    aggregate_contributions,
    original_feature_map,
)
from retention.explainability.schemas import FeatureContribution
from retention.explainability.service import (
    frozen_explanation_service,
    rank_factors,
)
from retention.features.preprocessing import make_preprocessor
from tests.integration.test_database import synthetic_customer


@pytest.fixture(scope="module")
def service():
    return frozen_explanation_service()


def test_raw_margin_additivity_and_calibrated_prediction(service):
    values = synthetic_customer()
    result = service.explain(values)
    predictor_result = service.predictor.predict(values).iloc[0]
    assert abs(result.additivity_residual) < 1e-4
    assert np.isclose(
        result.base_value
        + sum(
            item.raw_margin_contribution
            for item in result.transformed_feature_contributions
        ),
        result.raw_prediction_margin,
        atol=1e-4,
    )
    assert np.isclose(
        expit(result.raw_prediction_margin), result.raw_churn_probability, atol=1e-6
    )
    assert result.calibrated_churn_probability == predictor_result["churn_probability"]
    assert result.predicted_churn == predictor_result["predicted_churn"]
    assert result.threshold == predictor_result["threshold"]
    assert result.explained_output_space == "raw_xgboost_margin"
    assert result.feature_perturbation == "tree_path_dependent"
    assert len(result.original_feature_contributions) == 19
    assert len(result.transformed_feature_contributions) == 45
    assert np.isclose(
        sum(
            item.raw_margin_contribution
            for item in result.original_feature_contributions
        ),
        sum(
            item.raw_margin_contribution
            for item in result.transformed_feature_contributions
        ),
        atol=1e-8,
    )
    assert "customerID" not in result.to_dict()


def test_single_and_batched_explanations_agree(service):
    first = synthetic_customer()
    second = dict(first, tenure=13)
    one = service.explain(first)
    batch = service.explain_many([first, second])
    assert len(batch) == 2
    assert batch[0].raw_prediction_margin == one.raw_prediction_margin
    assert batch[0].original_feature_contributions == one.original_feature_contributions
    assert (
        batch[1].calibrated_churn_probability
        == service.predictor.predict(second).iloc[0]["churn_probability"]
    )
    assert (
        service.explain_many(pd.DataFrame([first]))[0].raw_prediction_margin
        == one.raw_prediction_margin
    )


def test_metadata_mapping_handles_reference_categories():
    first = synthetic_customer()
    second = dict(
        first,
        gender="Male",
        Partner="No",
        Dependents="Yes",
        PhoneService="No",
        MultipleLines="No phone service",
        InternetService="No",
        OnlineSecurity="No internet service",
        OnlineBackup="No internet service",
        DeviceProtection="No internet service",
        TechSupport="No internet service",
        StreamingTV="No internet service",
        StreamingMovies="No internet service",
        Contract="Month-to-month",
        PaperlessBilling="No",
        PaymentMethod="Mailed check",
    )
    frame = pd.DataFrame([first, second])
    preprocessor = make_preprocessor("baseline", "linear").fit(frame)
    mapping = original_feature_map(preprocessor, tuple(first))
    assert len(mapping) == preprocessor.transform(frame).shape[1]
    assert set(mapping) == set(first)
    row = np.arange(len(mapping), dtype=float)
    aggregated = aggregate_contributions(row, mapping, tuple(first))
    assert np.isclose(sum(aggregated.values()), row.sum())


def test_deterministic_ranking_and_strict_json(service):
    items = (
        FeatureContribution("A", 1, 0.5, "A increased score.", "service_or_billing"),
        FeatureContribution("B", 2, 0.5, "B increased score.", "service_or_billing"),
        FeatureContribution("C", 3, -0.2, "C reduced score.", "service_or_billing"),
        FeatureContribution(
            "D", 4, 0.0, "D did not change score.", "service_or_billing"
        ),
    )
    positive, negative, absolute = rank_factors(items, 5)
    assert [item.feature for item in positive] == ["A", "B"]
    assert [item.feature for item in negative] == ["C"]
    assert [item.feature for item in absolute] == ["A", "B", "C", "D"]
    result = service.explain(synthetic_customer())
    with pytest.raises(ValueError):
        replace(result, base_value=float("nan")).to_dict()
    assert result.from_dict(result.to_dict()).to_dict() == result.to_dict()


def test_invalid_input_and_nonfinite_contributions(service):
    values = synthetic_customer()
    with pytest.raises(ValueError):
        service.explain(dict(values, Contract="unknown"))
    with pytest.raises(ValueError):
        service.explain(dict(values, Churn="No"))
    with pytest.raises(ValueError):
        service.explain_many([values] * 101)
    with pytest.raises(ValueError):
        service.explain(values, top_k=0)
    with pytest.raises(ValueError):
        aggregate_contributions(np.array([float("nan")]), ("tenure",), ("tenure",))


def test_global_artifacts_use_synthetic_development_sample_and_cache(
    service, tmp_path, monkeypatch
):
    import hashlib
    from types import SimpleNamespace

    from retention.explainability import global_explanations as global_module

    first = synthetic_customer()
    second = dict(
        first, tenure=13, Contract="Month-to-month", InternetService="Fiber optic"
    )
    development = tmp_path / "development.parquet"
    pd.DataFrame([first, second] * 4).to_parquet(development)
    development_sha = hashlib.sha256(development.read_bytes()).hexdigest()
    predictor = SimpleNamespace(
        columns=service.predictor.columns,
        metadata={
            **service.predictor.metadata,
            "development_sha256": development_sha,
        },
    )
    fake_service = SimpleNamespace(
        predictor=predictor,
        model_sha256=service.model_sha256,
        fingerprint=service.fingerprint,
        explain_many=service.explain_many,
    )
    monkeypatch.setattr(
        global_module, "frozen_explanation_service", lambda: fake_service
    )
    output = tmp_path / "global"
    summary = global_module.generate_global_explanations(output, development)
    assert summary["sample_rows"] == 8
    assert len(summary["importance"]) == 19
    assert summary["fingerprints"]["development_sha256"] == development_sha
    assert all((output / name).is_file() for name in global_module.CHARTS)
    monkeypatch.setattr(
        global_module,
        "_save_bar",
        lambda *args: pytest.fail("matching artifacts should be reused"),
    )
    assert global_module.generate_global_explanations(output, development) == summary
