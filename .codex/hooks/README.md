# Codex data-science safety hooks

`../hooks.json` registers lifecycle hooks for local Codex sessions in this
trusted repository. `data_science_guard.py` uses only the Python standard
library and provides three layers:

1. `SessionStart` and `SubagentStart` remind every agent about the repository's
   data, leakage, evaluation, PII, artifact, and approval requirements.
2. `PreToolUse` blocks destructive shell commands, secret-file reads, edits to
   protected data/artifacts/policy files, edits to applied migrations, obvious
   secrets, and high-confidence leakage patterns before the tool runs.
3. `Stop` scans changed Python files for leakage and evaluation risks and asks
   the agent to correct findings before finishing.

After pulling or changing these files, open `/hooks` in Codex, inspect the
project hook, and trust its current hash. Project hooks run only for trusted
projects. Changes to the hook definition or script require review again.

These hooks are defense in depth, not a complete security boundary. Keep raw
data and model registries protected with database permissions, DVC/MLflow
controls, code review, and required CI checks. Some specialized or hosted tools
do not pass through local `PreToolUse` hooks.

Run the hook tests with:

```sh
.venv/bin/pytest tests/test_agent_hooks.py
```
