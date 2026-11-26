"""Integration tests for automated statistical-analysis reporting."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import retention.analysis.statistical_analysis as statistical_module
from retention.analysis.statistical_analysis import (
    StatisticalAnalysisError,
    run_statistical_analysis,
)


@pytest.fixture
def statistical_dataset() -> pd.DataFrame:
    """Return a varied, structurally consistent development population."""
    rows = []
    addons = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for index in range(120):
        phone = "No" if index % 10 == 0 else "Yes"
        internet = (
            "No" if index % 5 == 0 else ("DSL" if index % 2 == 0 else "Fiber optic")
        )
        row = {
            "customerID": f"S-{index:04d}",
            "gender": "Male" if index % 2 else "Female",
            "SeniorCitizen": index % 2,
            "Partner": "Yes" if index % 3 else "No",
            "Dependents": "Yes" if index % 4 else "No",
            "tenure": index % 73,
            "PhoneService": phone,
            "MultipleLines": "No phone service"
            if phone == "No"
            else ("Yes" if index % 3 else "No"),
            "InternetService": internet,
            "Contract": ["Month-to-month", "One year", "Two year"][index % 3],
            "PaperlessBilling": "Yes" if index % 2 else "No",
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ][index % 4],
            "MonthlyCharges": 20.0 + index % 90,
            "TotalCharges": (20.0 + index % 90) * (index % 73),
            "Churn": "Yes" if index % 4 == 0 else "No",
        }
        for offset, addon in enumerate(addons):
            row[addon] = (
                "No internet service"
                if internet == "No"
                else ("Yes" if (index + offset) % 3 else "No")
            )
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def statistical_config(tmp_path: Path, statistical_dataset: pd.DataFrame) -> Path:
    """Write an isolated statistical-analysis configuration and input."""
    development = tmp_path / "development.parquet"
    statistical_dataset.to_parquet(development, index=False)
    categorical = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    config = {
        "project": {"random_state": 42},
        "data": {
            "development_path": str(development),
            "test_path": str(tmp_path / "test.parquet"),
            "validated_path": str(tmp_path / "validated.parquet"),
            "target_column": "Churn",
            "id_column": "customerID",
        },
        "eda": {"categorical_features": categorical},
        "statistical_analysis": {
            "significance_level": 0.05,
            "fdr_threshold": 0.05,
            "confidence_level": 0.95,
            "registered_predictors": {
                "numerical": {
                    "HYP-01": "tenure",
                    "HYP-02": "MonthlyCharges",
                    "HYP-03": "TotalCharges",
                },
                "categorical": {
                    "HYP-04": "Contract",
                    "HYP-05": "InternetService",
                    "HYP-06": "PaymentMethod",
                    "HYP-07": "TechSupport",
                    "HYP-08": "OnlineSecurity",
                    "HYP-09": "PaperlessBilling",
                    "HYP-10a": "gender",
                    "HYP-10b": "SeniorCitizen",
                    "HYP-10c": "Partner",
                    "HYP-10d": "Dependents",
                },
                "confidence_segments": ["Contract"],
            },
            "minimum_segment_size": 10,
            "vif_screening_thresholds": [5.0, 10.0],
            "output_directory": str(tmp_path / "statistical_analysis"),
        },
    }
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_complete_statistical_analysis(
    statistical_config: Path, tmp_path: Path
) -> None:
    summary = run_statistical_analysis(statistical_config)
    output = tmp_path / "statistical_analysis"
    expected = {
        "summary.json",
        "numerical_hypothesis_tests.csv",
        "categorical_hypothesis_tests.csv",
        "adjusted_hypothesis_tests.csv",
        "confidence_intervals.csv",
        "multicollinearity.csv",
        "report.html",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert summary["hypothesis_families"]["primary_feature_churn"]["test_count"] == 13
    assert (
        summary["hypothesis_families"]["complementary_numerical_ranks"]["test_count"]
        == 3
    )
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["dataset_dimensions"] == {"columns": 21, "rows": 120}


def test_statistical_outputs_are_deterministic(
    statistical_config: Path, tmp_path: Path
) -> None:
    run_statistical_analysis(statistical_config)
    first = _tree_hashes(tmp_path / "statistical_analysis")
    run_statistical_analysis(statistical_config)
    assert _tree_hashes(tmp_path / "statistical_analysis") == first


def test_statistical_analysis_reads_only_development(
    statistical_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pd.read_parquet
    accessed: list[str] = []

    def guarded(path: str | Path, *args: object, **kwargs: object) -> pd.DataFrame:
        accessed.append(Path(path).name)
        if Path(path).name != "development.parquet":
            raise AssertionError("held-out data access")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)
    run_statistical_analysis(statistical_config)
    assert accessed == ["development.parquet"]


def test_statistical_analysis_rejects_unsafe_paths(
    statistical_config: Path, tmp_path: Path
) -> None:
    config = yaml.safe_load(statistical_config.read_text())
    config["statistical_analysis"]["output_directory"] = str(tmp_path)
    statistical_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(StatisticalAnalysisError, match="parent directories"):
        run_statistical_analysis(statistical_config)


def test_statistical_analysis_rejects_test_input(
    statistical_config: Path,
) -> None:
    config = yaml.safe_load(statistical_config.read_text())
    config["data"]["development_path"] = config["data"]["test_path"]
    statistical_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(StatisticalAnalysisError):
        run_statistical_analysis(statistical_config)


def test_statistical_analysis_rejects_missing_input(
    statistical_config: Path, tmp_path: Path
) -> None:
    (tmp_path / "development.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        run_statistical_analysis(statistical_config)


@pytest.mark.parametrize(
    ("key", "value"),
    [("significance_level", 0.0), ("minimum_segment_size", 0)],
)
def test_statistical_analysis_rejects_invalid_configuration(
    statistical_config: Path, key: str, value: float
) -> None:
    config = yaml.safe_load(statistical_config.read_text())
    config["statistical_analysis"][key] = value
    statistical_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(StatisticalAnalysisError):
        run_statistical_analysis(statistical_config)


def test_failed_publication_preserves_current_report(
    statistical_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_statistical_analysis(statistical_config)
    output = tmp_path / "statistical_analysis"
    before = (output / "summary.json").read_bytes()
    monkeypatch.setattr(
        statistical_module,
        "write_statistical_html",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("HTML failure")),
    )
    with pytest.raises(RuntimeError, match="HTML failure"):
        run_statistical_analysis(statistical_config)
    assert (output / "summary.json").read_bytes() == before
    assert not (tmp_path / "statistical_analysis.staging").exists()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
