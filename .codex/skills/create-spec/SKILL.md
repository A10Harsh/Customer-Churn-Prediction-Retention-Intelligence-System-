---
name: create-spec
description: Create a scoped implementation specification for the next authorized churn-project phase or feature, and optionally create a local feature branch when the user explicitly requests one. Do not use for implementing the specification.
---

# Create Project Specification

Create an evidence-based specification that another agent can implement without
guessing scope, data contracts, acceptance criteria, or safety boundaries.

## Inputs

Derive these values from the user's request:

- step number, zero-padded to two digits;
- feature title in title case;
- feature slug in lowercase kebab-case, limited to letters, digits, and hyphens;
- whether a local feature branch was explicitly requested.

Ask one concise clarification question only when the step or feature cannot be
identified reliably. Do not invent a phase, dataset, schema, or deliverable.

## Establish repository context

Before drafting:

1. Read `AGENTS.md` and `.codex/spec.md` completely.
2. Inspect `git status --short`, the current branch, and relevant diffs. Preserve
   unrelated user changes; a dirty worktree is not automatically a blocker.
3. Inspect existing specifications under `.codex/spec/` and do not duplicate or
   overwrite them.
4. Inspect the relevant source, tests, configuration, pipelines, database SQL,
   notebooks, and documentation for the requested phase.
5. Inspect data only with non-destructive commands. Do not read `.env`, expose
   row-level identifiers, or modify source data.
6. Confirm that prerequisite phases and their actual checks are complete. If a
   prerequisite is incomplete, write it as a blocking dependency rather than
   pretending the requested phase is ready.

Use facts measured from repository artifacts. Do not copy dataset dimensions,
class balance, metrics, drift values, or model results from memory or plans.

## Local branch policy

Create a branch only when the user explicitly asks for one.

- Use `feature/<step-number>-<feature-slug>`.
- Check local branch names first; append `-01`, `-02`, and so on when needed.
- If unrelated working-tree changes make switching unsafe, stop and ask the user
  how to handle them. Never stash, discard, commit, or reset user work yourself.
- Use a non-destructive local branch command such as
  `git switch -c <branch-name>`.
- Never pull, fetch, push, merge, rebase, or contact a remote unless separately
  requested and authorized.

If no branch was requested, remain on the current branch.

## Specification path

Write the specification to:

```text
.codex/spec/<step-number>-<feature-slug>.md
```

Do not overwrite an existing specification. If that path exists, report it and
ask whether the user wants a review or a separately scoped revision.

## Required specification structure

Use only sections relevant to the requested work, but always include the phase,
scope, dependencies, implementation plan, validation, and definition of done.

```markdown
# Spec: <Feature Title>

## Metadata

- Step: <number>
- Status: Proposed
- Branch: <branch name or "Not requested">
- Governing requirements: `AGENTS.md`, `.codex/spec.md`

## Objective

Describe the outcome, why it is needed now, and the concrete deliverables.

## Dependencies and entry gate

List prerequisite phases, artifacts, and checks. Distinguish verified evidence
from missing or blocked prerequisites.

## Current-state evidence

Record the files, commands, data versions, and actual observations used to
prepare the specification. Do not include secrets, raw records, or invented
metrics.

## Scope

### In scope

List the authorized implementation work.

### Out of scope

List adjacent phases, destructive actions, external side effects, and approval-
gated operations that are not authorized by this specification.

## Data and artifact contracts

Document existing input locations, formats, schemas, target and identifier
handling, expected outputs, versioning, and validation requirements. State "No
new data requirements" when applicable.

## Implementation plan

Describe the production modules, SQL, configuration, pipelines, notebooks,
tests, reports, and documentation that must be created or changed. Keep reusable
logic in `src/`; notebooks remain exploratory callers.

## Pipeline and lineage

Define required scikit-learn, DVC, Prefect, MLflow, or serving pipeline changes
for this phase. Declare stage dependencies, parameters, outputs, metrics, and
rerun behavior where applicable.

## Files to create

List each new file and its responsibility.

## Files to modify

List each existing file and the intended change. Do not include unrelated
refactors.

## Dependencies

List new packages with justification and `pyproject.toml` changes, or state "No
new dependencies".

## Tests and validation

List unit, integration, leakage, parity, schema, serialization, reproducibility,
and smoke checks relevant to the phase, including commands and expected
observable outcomes.

## Safety and approval gates

List protected data/artifacts, privacy constraints, and operations that require
explicit approval.

## Definition of done

Provide a testable checklist based on actual commands and artifacts. Include
Ruff and pytest, and treat no tests collected as inconclusive.

## Handoff

State the artifacts and evidence the next phase or agent will consume.
```

## Project invariants

Include the relevant invariants below in every specification that touches them:

### Data and preparation

- Raw customer data is immutable, excluded from Git, and tracked through DVC
  metadata only.
- Map `Churn` from `Yes/No` to `1/0` only in staging.
- Preserve blank `TotalCharges` values as missing numeric data until imputation
  inside the training pipeline; never silently drop rows.
- Exclude `Churn` and `customerID` from model features and LLM inputs/logs.
- Split before fitting: stratified `70/15/15`, configured seed `42`.
- Fit learned preprocessing on training data only and use one shared pipeline
  for training, batch scoring, and serving.

### Model development

- Log the untuned logistic-regression baseline before Random Forest, XGBoost,
  or direct PyTorch ANN challengers.
- Never use `MLPClassifier`; the ANN must use the approved serializable PyTorch
  sklearn-compatible adapter.
- Tune with training-fold cross-validation only. Select by validation PR-AUC.
- Calibrate and choose threshold evidence on validation only.
- Read the test set once for a frozen final candidate and never tune from it.
- Report only real PR-AUC, ROC-AUC, precision, recall, F1, and calibration
  evidence produced by executed runs.
- Use MLflow experiment `churn-retention` and registered model name
  `churn-classifier`.

### Reproducibility and operations

- Declare applicable stages, dependencies, parameters, outputs, and metrics in
  `dvc.yaml`; validate ordering with `dvc dag` and reproduction with `dvc repro`.
- Preserve DVC, MLflow, registered-model, and migration artifacts.
- Use fake external clients in tests and CI; never require live secrets.
- Require explicit approval before threshold changes, champion promotion,
  destructive database operations, deployment, remote pushes, or paid actions.

## Completion report

After writing the specification, report:

```text
Branch:    <branch name or "unchanged">
Spec file: .codex/spec/<step-number>-<feature-slug>.md
Title:     <feature title>
Gate:      <ready or blocked, with the blocking dependency>
```

Do not implement the specification in the same task unless the user explicitly
requests both specification and implementation.
