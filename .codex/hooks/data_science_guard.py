#!/usr/bin/env python3
"""Codex hook guardrails for destructive actions and ML development mistakes.

The hook is deliberately dependency-free so it can run before the project's
virtual environment is available. It is a defense-in-depth control; CI,
reviews, tests, DVC, database permissions, and MLflow permissions remain the
authoritative enforcement layers.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SAFETY_CONTEXT = "\n".join(
    (
        "Repository safety policy:",
        "- Never delete or overwrite data, DVC/MLflow artifacts, registered ",
        "  models, or applied migrations.",
        "- Split before fitting; fit preprocessing on training data only; ",
        "  exclude Churn and customerID.",
        "- Never tune thresholds/hyperparameters on the test set. Report ",
        "  ROC-AUC, PR-AUC, precision, and recall from real runs.",
        "- Never expose .env values or send PII/customer IDs to an LLM.",
        "- Model promotion, threshold changes, deployments, remote pushes, ",
        "  and cost-incurring actions require explicit approval.",
        "- Run Ruff and pytest and report actual results before completion.",
    )
)

PROTECTED_PATCH_PATHS = (
    ".codex/hooks.json",
    ".codex/hooks/",
    ".env",
    "AGENTS.md",
    "data/",
    "mlruns/",
    "artifacts/",
    "model_artifacts/",
    "dvc.lock",
)

DESTRUCTIVE_COMMAND_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(^|[;&|]\s*)(sudo\s+)?(rm|rmdir|unlink|shred)\b", re.I),
        "filesystem deletion is blocked; obtain explicit approval and use a "
        "reviewed, exact target",
    ),
    (
        re.compile(r"\bfind\b[^\n]*(?:-delete|-exec\s+(?:rm|shred))\b", re.I),
        "recursive deletion through find is blocked",
    ),
    (
        re.compile(r"\bgit\s+(?:reset\s+--hard|clean\b|checkout\s+--|restore\b)", re.I),
        "destructive Git restoration/cleanup is blocked",
    ),
    (
        re.compile(r"\b(?:dvc\s+(?:remove|destroy|gc)|mlflow\s+gc)\b", re.I),
        "deleting versioned data or tracked model artifacts is blocked",
    ),
    (
        re.compile(
            r"\bdocker\b[^\n]*(?:\bdown\b[^\n]*\s-v\b|\bvolume\s+(?:rm|prune)\b)",
            re.I,
        ),
        "Docker volume deletion is blocked because it can destroy "
        "PostgreSQL/MLflow data",
    ),
    (
        re.compile(
            r"\b(?:drop\s+(?:table|schema|database)|truncate\s+(?:table\s+)?|"
            r"delete\s+from)\b",
            re.I,
        ),
        "destructive SQL is blocked",
    ),
    (
        re.compile(r"\b(?:insert\s+into|update|alter\s+table)\s+(?:raw\.)", re.I),
        "the raw database schema is immutable",
    ),
    (
        re.compile(
            r"\bgit\s+push\b|\bterraform\s+(?:apply|destroy)\b|"
            r"\bkubectl\s+(?:apply|delete)\b",
            re.I,
        ),
        "remote, deployment, or cost-impacting actions require explicit user approval",
    ),
    (
        re.compile(
            r"\b(?:transition_model_version_stage|set_registered_model_alias)\b",
            re.I,
        ),
        "model registry promotion requires a logged comparison and explicit approval",
    ),
)

SECRET_READ_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:cat|head|tail|less|more|sed|awk|grep|rg|strings|source|\.)\b"
    r"[^\n]*(?:^|[/\s])\.env(?![\w.-])",
    re.I,
)

SECRET_ADDITION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "private key material",
    ),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "API key-like value"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key"),
    (
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|passwd|access[_-]?token)\b\s*[:=]\s*['\"][^'\"${}]{8,}['\"]"
        ),
        "hard-coded credential",
    ),
)

LEAKAGE_ADDITION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\.(?:fit|fit_transform)\s*\(\s*"
            r"(?:X_test|x_test|test_df|validation_df|val_df)\b"
        ),
        "fitting a model or transformer on validation/test data causes leakage",
    ),
    (
        re.compile(
            r"(?i)(?:study\.optimize|GridSearchCV|RandomizedSearchCV)[^\n]*"
            r"(?:X_test|y_test|test_df)"
        ),
        "test data must not be used for hyperparameter tuning",
    ),
    (
        re.compile(r"(?i)threshold[^\n]*(?:X_test|y_test|test_df)"),
        "the decision threshold must be selected on validation data, not test data",
    ),
    (
        re.compile(
            r"(?i)(?:features?|predictors?|feature_columns)[^\n]*"
            r"['\"](?:Churn|customerID)['\"]"
        ),
        "Churn and customerID cannot be model features",
    ),
    (
        re.compile(
            r"(?i)(?:pd\.)?concat\s*\(\s*\[[^\]]*"
            r"(?:X_train|train_df)[^\]]*(?:X_test|test_df)"
        ),
        "combining train and test data before fitting creates leakage risk",
    ),
)


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":")))


def _deny(reason: str) -> None:
    _emit(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Data-science safety hook: {reason}.",
            }
        }
    )


def _command_from_event(event: dict[str, Any]) -> str:
    tool_input = event.get("tool_input", {})
    if isinstance(tool_input, str):
        return tool_input
    if not isinstance(tool_input, dict):
        return ""
    for key in ("command", "cmd", "patch"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return ""


def check_shell_command(command: str) -> str | None:
    """Return a blocking reason when a shell command violates project policy."""
    if SECRET_READ_RE.search(command):
        return (
            "reading local .env secrets is blocked; use .env.example variable "
            "names only"
        )

    for pattern, reason in DESTRUCTIVE_COMMAND_RULES:
        if pattern.search(command):
            return reason
    return None


def _patch_operations(patch: str) -> list[tuple[str, str]]:
    return [
        (match.group(1), match.group(2).strip())
        for match in re.finditer(
            r"^\*\*\* (Add|Update|Delete) File: (.+)$", patch, re.M
        )
    ]


def _is_protected_path(path: str) -> bool:
    normalized = path.removeprefix("./")
    return any(
        normalized == protected.rstrip("/") or normalized.startswith(protected)
        for protected in PROTECTED_PATCH_PATHS
    )


def _added_lines(patch: str) -> str:
    return "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def check_patch(patch: str) -> str | None:
    """Return a blocking reason for a dangerous or leakage-prone patch."""
    for operation, path in _patch_operations(patch):
        if path == ".env" or path.startswith(".env.") and path != ".env.example":
            return f"agents may not {operation.lower()} local secret file {path}"
        if _is_protected_path(path):
            return (
                f"{operation.lower()} of protected path {path} requires manual review"
            )
        if path.startswith("database/migrations/") and operation != "Add":
            return (
                "applied migrations are immutable; add a new numbered migration instead"
            )

    additions = _added_lines(patch)
    for pattern, label in SECRET_ADDITION_RULES:
        if pattern.search(additions):
            return f"the patch appears to add {label}"
    for pattern, reason in LEAKAGE_ADDITION_RULES:
        if pattern.search(additions):
            return reason
    return None


def scan_python_text(path: str, text: str) -> list[str]:
    """Find high-confidence ML development risks in Python source text."""
    findings: list[str] = []
    lowered = text.lower()

    if re.search(
        r"\.fit(?:_transform)?\s*\(\s*(?:x_test|test_df|validation_df|val_df)\b",
        text,
        re.I,
    ):
        findings.append(f"{path}: fits on validation/test data")

    if (
        "optuna" in lowered
        or "gridsearchcv" in lowered
        or "randomizedsearchcv" in lowered
    ) and re.search(
        r"(?s)(?:objective|study\.optimize|GridSearchCV|RandomizedSearchCV).{0,2000}\by_test\b",
        text,
        re.I,
    ):
        findings.append(f"{path}: tuning code appears to reference y_test")

    if "accuracy_score" in text and not all(
        metric in text
        for metric in ("precision_score", "recall_score", "average_precision_score")
    ):
        findings.append(
            f"{path}: accuracy is reported without precision, recall, and "
            "PR-AUC (average precision)"
        )

    if "roc_auc_score" in text and "average_precision_score" not in text:
        findings.append(
            f"{path}: ROC-AUC is reported without PR-AUC (average precision)"
        )

    if "train_test_split" in text and "random_state" not in text:
        findings.append(f"{path}: data split has no reproducible random_state")

    preprocessors = ("StandardScaler", "OneHotEncoder", "SimpleImputer")
    if any(name in text for name in preprocessors) and not re.search(
        r"\b(?:Pipeline|make_pipeline)\b", text
    ):
        findings.append(f"{path}: preprocessing is not visibly contained in a Pipeline")

    if re.search(
        r"(?i)(?:features?|predictors?|feature_columns)[^\n]*"
        r"['\"](?:Churn|customerID)['\"]",
        text,
    ):
        findings.append(f"{path}: target or identifier appears in model features")

    return findings


def _changed_python_files(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    paths: list[Path] = []
    entries = result.stdout.split("\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        status = entry[:2]
        relative = entry[3:]
        if "R" in status and index < len(entries):
            relative = entries[index]
            index += 1
        candidate = repo_root / relative
        source_roots = {"src", "pipelines"}
        if (
            candidate.suffix == ".py"
            and candidate.is_file()
            and Path(relative).parts[0] in source_roots
        ):
            paths.append(candidate)
    return paths


def audit_changed_python(repo_root: Path) -> list[str]:
    """Audit changed Python files for high-confidence ML safety problems."""
    findings: list[str] = []
    for path in _changed_python_files(repo_root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        findings.extend(scan_python_text(str(path.relative_to(repo_root)), text))
    return findings


def _handle_context_event(event_name: str) -> None:
    _emit(
        {
            "hookSpecificOutput": {
                "hookEventName": event_name,
                "additionalContext": SAFETY_CONTEXT,
            }
        }
    )


def _handle_pre_tool(event: dict[str, Any]) -> None:
    tool_name = str(event.get("tool_name", ""))
    content = _command_from_event(event)
    if tool_name == "Bash":
        reason = check_shell_command(content)
    elif tool_name in {"apply_patch", "Edit", "Write"}:
        reason = check_patch(content)
    else:
        reason = None

    if reason:
        _deny(reason)


def _handle_stop(event: dict[str, Any]) -> None:
    if event.get("stop_hook_active"):
        _emit({"continue": True})
        return

    repo_root = Path(str(event.get("cwd", "."))).resolve()
    findings = audit_changed_python(repo_root)
    if findings:
        reason = "Resolve these ML safety findings before finishing:\n- " + "\n- ".join(
            findings[:12]
        )
        _emit({"decision": "block", "reason": reason})
    else:
        _emit({"continue": True})


def main() -> int:
    """Read one hook event from stdin and emit the Codex hook response."""
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return 0

    event_name = str(event.get("hook_event_name", ""))
    if event_name in {"SessionStart", "SubagentStart"}:
        _handle_context_event(event_name)
    elif event_name == "PreToolUse":
        _handle_pre_tool(event)
    elif event_name == "Stop":
        _handle_stop(event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
