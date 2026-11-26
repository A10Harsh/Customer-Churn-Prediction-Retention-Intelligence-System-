"""Unit tests for VIF and matrix diagnostics."""

import numpy as np
import pandas as pd
import pytest

from retention.analysis.multicollinearity import (
    calculate_vif,
    matrix_diagnostics,
    reference_coded_design,
)


def test_independent_and_exact_vif() -> None:
    independent = pd.DataFrame(
        {"x1": [-1.0, -1.0, 1.0, 1.0], "x2": [-1.0, 1.0, -1.0, 1.0]}
    )
    assert np.allclose(calculate_vif(independent)["vif"], 1.0)
    dependent = independent.assign(x3=lambda frame: frame["x1"] + frame["x2"])
    result = calculate_vif(dependent)
    assert np.isinf(result["vif"]).all()
    assert result["dependency_type"].eq("exact_linear_dependency").all()


def test_rank_deficiency_and_condition_number() -> None:
    design = pd.DataFrame(
        {
            "x1": [-1.0, -1.0, 1.0, 1.0],
            "x2": [-1.0, 1.0, -1.0, 1.0],
            "x3": [-2.0, 0.0, 0.0, 2.0],
        }
    )
    result = matrix_diagnostics(design)
    assert result["rank_deficiency"] == 1
    assert np.isinf(result["condition_number"])


def test_reference_encoding_records_reference_and_drops_one_level() -> None:
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "group": ["A", "B", "A"]})
    design, references = reference_coded_design(data, ["x"], ["group"])
    assert references == {"group": "A"}
    assert list(design) == ["x", "group_B"]


def test_multicollinearity_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        calculate_vif(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        calculate_vif(pd.DataFrame({"x": [1.0, np.nan], "y": [2.0, 3.0]}))
    with pytest.raises(ValueError):
        calculate_vif(
            pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]}),
            exact_tolerance=0,
        )
    with pytest.raises(ValueError):
        matrix_diagnostics(pd.DataFrame({"x": [1.0, np.nan]}))
    with pytest.raises(ValueError):
        matrix_diagnostics(pd.DataFrame())
    with pytest.raises(ValueError):
        reference_coded_design(pd.DataFrame({"x": [1.0], "group": ["A"]}), ["x"], ["x"])


def test_constant_vif_and_constant_matrix_are_explicit() -> None:
    design = pd.DataFrame({"constant": [1.0] * 4, "varying": [1.0, 2.0, 3.0, 4.0]})
    result = calculate_vif(design).set_index("predictor")
    assert result.loc["constant", "dependency_type"] == "constant"
    assert np.isinf(result.loc["constant", "vif"])
    diagnostics = matrix_diagnostics(pd.DataFrame({"constant": [1.0] * 4}))
    assert diagnostics["constant_columns"] == ["constant"]
    assert diagnostics["rank_deficiency"] == 0


def test_reference_encoding_rejects_missing_and_nonfinite_columns() -> None:
    with pytest.raises(ValueError, match="missing"):
        reference_coded_design(pd.DataFrame({"x": [1.0]}), ["x"], ["group"])
    with pytest.raises(ValueError, match="non-finite"):
        reference_coded_design(
            pd.DataFrame({"x": [1.0, np.nan], "group": ["A", "B"]}),
            ["x"],
            ["group"],
        )
