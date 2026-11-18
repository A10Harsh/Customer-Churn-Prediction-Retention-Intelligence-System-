"""Tests for project-local Codex data-science safety hooks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    path = Path(__file__).parents[1] / ".codex" / "hooks" / "data_science_guard.py"
    spec = importlib.util.spec_from_file_location("data_science_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUARD = _load_guard()


def test_blocks_destructive_commands() -> None:
    assert GUARD.check_shell_command("rm -rf data/raw") is not None
    assert GUARD.check_shell_command("psql -c 'TRUNCATE raw.customers'") is not None
    assert GUARD.check_shell_command("git reset --hard HEAD") is not None


def test_allows_normal_validation_commands() -> None:
    assert GUARD.check_shell_command(".venv/bin/ruff check .") is None
    assert GUARD.check_shell_command(".venv/bin/pytest") is None


def test_blocks_env_secret_reads_but_allows_example() -> None:
    assert GUARD.check_shell_command("cat .env") is not None
    assert GUARD.check_shell_command("cat .env.example") is None
    assert GUARD.check_shell_command("cat .env .env.example") is not None


def test_blocks_protected_file_and_migration_edits() -> None:
    protected = "*** Begin Patch\n*** Update File: AGENTS.md\n+x\n*** End Patch"
    migration = (
        "*** Begin Patch\n"
        "*** Update File: database/migrations/001_init.sql\n"
        "+x\n*** End Patch"
    )
    assert GUARD.check_patch(protected) is not None
    assert GUARD.check_patch(migration) is not None


def test_allows_new_migration() -> None:
    patch = (
        "*** Begin Patch\n"
        "*** Add File: database/migrations/002_index.sql\n"
        "+x\n*** End Patch"
    )
    assert GUARD.check_patch(patch) is None


def test_blocks_test_set_fitting_and_identifier_features() -> None:
    fit_patch = (
        "*** Begin Patch\n*** Update File: src/models/train.py\n"
        "+.fit(X_test)\n*** End Patch"
    )
    feature_patch = (
        "*** Begin Patch\n*** Update File: src/models/train.py\n"
        "+feature_columns = ['tenure', 'customerID']\n*** End Patch"
    )
    assert GUARD.check_patch(fit_patch) is not None
    assert GUARD.check_patch(feature_patch) is not None


def test_flags_incomplete_imbalanced_classification_metrics() -> None:
    findings = GUARD.scan_python_text(
        "src/models/evaluate.py",
        "from sklearn.metrics import accuracy_score, roc_auc_score\n",
    )
    assert any("precision" in finding for finding in findings)
    assert any("PR-AUC" in finding for finding in findings)


def test_accepts_complete_metrics_and_reproducible_split() -> None:
    source = """
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
train_test_split(X, y, random_state=42)
"""
    assert GUARD.scan_python_text("src/models/evaluate.py", source) == []
