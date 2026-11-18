# Machine-Learning Data Preparation Skill Library

This repository-local library gives coding agents an executable path from raw tabular data to validated, model-ready train, validation, and test artifacts. It is deliberately modular: invoke only the skill needed for the current decision, or follow the end-to-end workflow.

## Operating contract

- Preserve immutable raw inputs and write derived artifacts to configured interim or processed locations.
- Identify the target, entity key, prediction timestamp, and observation cutoff before analysis.
- Split before learning statistics. Fit imputers, encoders, scalers, samplers, selectors, and reducers on training folds only.
- Exclude targets, post-outcome fields, direct identifiers, and unsupported PII from model features.
- Use deterministic seeds and persist split indices, schemas, fitted pipelines, and data-quality reports.
- Prefer `Pipeline` and `ColumnTransformer`; place imbalance samplers inside an `imblearn.pipeline.Pipeline`.
- Treat plots and statistics as evidence, not automatic transformation commands. Record decisions and rejected alternatives.
- Do not fabricate quality, drift, or model-readiness metrics; generate them from the actual dataset.

## Layout

- `data-understanding/`: ingestion, profiling, schema and type inference, quality baselines.
- `eda/`: univariate through target-aware analysis.
- `data-cleaning/`: train-safe correction policies for missing, duplicate, outlier, inconsistent, and invalid values.
- `feature-engineering/`: typed feature construction, transforms, selection, and reduction.
- `preprocessing/`: splitting, resampling, cross-validation, and composable pipelines.
- `data-validation/`: leakage, feature contracts, consistency, drift, and readiness gates.
- [SKILL_INDEX.md](SKILL_INDEX.md): routing table for every skill.
- [WORKFLOW.md](WORKFLOW.md): orchestration, artifacts, and stop conditions.

## Expected environment

Examples target Python 3.11 and use pandas, NumPy, SciPy, scikit-learn, seaborn, and imbalanced-learn where appropriate. They are patterns to adapt into `src/`; production paths and thresholds belong in `configs/*.yaml`. Install packages only when the active project requires them and after explaining each new dependency.

## How to invoke

Ask the agent to use a skill by its frontmatter name, for example `use split-datasets` or `use detect-data-leakage`. For a complete preparation request, start with `ingest-data`, follow [WORKFLOW.md](WORKFLOW.md), and stop at any failed gate rather than silently coercing data.
