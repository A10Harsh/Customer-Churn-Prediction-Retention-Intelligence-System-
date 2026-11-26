"""Integration tests for complete automated EDA artifact generation."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import retention.analysis.eda as eda_module
from retention.analysis.eda import EDAError, run_eda


@pytest.fixture
def eda_dataset() -> pd.DataFrame:
    """Return a varied schema-compatible development population."""
    rows = []
    for i in range(80):
        internet = "No" if i % 4 == 0 else ("DSL" if i % 2 else "Fiber optic")
        no_internet = (
            "No internet service" if internet == "No" else ("Yes" if i % 3 else "No")
        )
        rows.append(
            {
                "customerID": f"C-{i:04d}",
                "gender": "Male" if i % 2 else "Female",
                "SeniorCitizen": i % 2,
                "Partner": "Yes" if i % 3 else "No",
                "Dependents": "Yes" if i % 5 else "No",
                "tenure": i % 73,
                "PhoneService": "Yes",
                "MultipleLines": "Yes" if i % 3 else "No",
                "InternetService": internet,
                "OnlineSecurity": no_internet,
                "OnlineBackup": no_internet,
                "DeviceProtection": no_internet,
                "TechSupport": no_internet,
                "StreamingTV": no_internet,
                "StreamingMovies": no_internet,
                "Contract": ["Month-to-month", "One year", "Two year"][i % 3],
                "PaperlessBilling": "Yes" if i % 2 else "No",
                "PaymentMethod": [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                ][i % 3],
                "MonthlyCharges": 20.0 + i,
                "TotalCharges": (20.0 + i) * (i % 73),
                "Churn": "Yes" if i % 4 == 0 else "No",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def eda_config(tmp_path: Path, eda_dataset: pd.DataFrame) -> Path:
    """Write an isolated development input and configuration."""
    development = tmp_path / "development.parquet"
    eda_dataset.to_parquet(development, index=False)
    config = {
        "project": {"random_state": 42},
        "data": {
            "development_path": str(development),
            "test_path": str(tmp_path / "test.parquet"),
            "validated_path": str(tmp_path / "validated.parquet"),
            "target_column": "Churn",
            "id_column": "customerID",
        },
        "eda": {
            "report_directory": str(tmp_path / "eda"),
            "numerical_features": ["tenure", "MonthlyCharges", "TotalCharges"],
            "categorical_features": [
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
            ],
            "rare_category_threshold": 0.01,
            "near_constant_dominance_threshold": 0.95,
            "iqr_multiplier": 1.5,
            "z_score_threshold": 3.0,
            "modified_z_score_threshold": 3.5,
            "percentile_bounds": [0.01, 0.99],
            "strong_association_thresholds": {
                "numerical": 0.8,
                "categorical": 0.4,
                "target": 0.2,
            },
            "minimum_segment_sample_size": 30,
            "tenure_boundaries": [-0.001, 6, 12, 24, 48, 72],
            "mutual_information": {"n_neighbors": 3},
            "figures": {"dpi": 60, "width": 5, "height": 3},
        },
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_complete_eda_orchestration(eda_config: Path, tmp_path: Path) -> None:
    summary = run_eda(eda_config)
    output = tmp_path / "eda"
    assert summary["schema_version"] == "1.0.0"
    assert (output / "summary.json").is_file() and (output / "report.html").is_file()
    assert len(list((output / "tables").glob("*.csv"))) >= 15
    assert len(list((output / "figures").glob("*.png"))) == 11
    assert (
        json.loads((output / "summary.json").read_text())["dataset_dimensions"]["rows"]
        == 80
    )


def test_outputs_are_reproducible(eda_config: Path, tmp_path: Path) -> None:
    run_eda(eda_config)
    first = _tree_hashes(tmp_path / "eda")
    run_eda(eda_config)
    assert _tree_hashes(tmp_path / "eda") == first


def test_input_is_preserved(eda_config: Path, tmp_path: Path) -> None:
    source = tmp_path / "development.parquet"
    before = source.read_bytes()
    run_eda(eda_config)
    assert source.read_bytes() == before


def test_missing_input_fails(eda_config: Path, tmp_path: Path) -> None:
    (tmp_path / "development.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        run_eda(eda_config)
    assert not (tmp_path / "eda").exists()


def test_failed_generation_preserves_current_report(
    eda_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_eda(eda_config)
    summary_before = (tmp_path / "eda" / "summary.json").read_bytes()
    monkeypatch.setattr(
        eda_module,
        "generate_figures",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plot failure")),
    )
    with pytest.raises(RuntimeError, match="plot failure"):
        run_eda(eda_config)
    assert (tmp_path / "eda" / "summary.json").read_bytes() == summary_before
    assert not (tmp_path / "eda.staging").exists()


def test_only_development_parquet_is_read(
    eda_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pd.read_parquet
    accessed = []

    def guarded(path: str | Path, *args: object, **kwargs: object) -> pd.DataFrame:
        accessed.append(Path(path).name)
        if Path(path).name != "development.parquet":
            raise AssertionError("held-out data access")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)
    run_eda(eda_config)
    assert accessed == ["development.parquet"]


def test_rejects_test_path_as_input(eda_config: Path, tmp_path: Path) -> None:
    config = yaml.safe_load(eda_config.read_text())
    config["data"]["development_path"] = config["data"]["test_path"]
    eda_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(EDAError):
        run_eda(eda_config)


def test_rejects_report_directory_containing_input(
    eda_config: Path, tmp_path: Path
) -> None:
    """Prevent atomic publishing from replacing a broad input parent directory."""
    config = yaml.safe_load(eda_config.read_text())
    config["eda"]["report_directory"] = str(tmp_path)
    eda_config.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(EDAError, match="parent directories"):
        run_eda(eda_config)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
