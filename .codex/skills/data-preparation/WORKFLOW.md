# Workflow Orchestration Guide

## 1. Establish the prediction contract

Record the unit of prediction, target definition, positive class, prediction time, observation window, outcome window, entity/group keys, and acceptable data latency. Stop if the target can only be known using post-prediction information.

## 2. Understand immutable inputs

Run `ingest-data`, `infer-data-types`, `detect-schema`, `profile-data`, and `assess-data-quality`. Produce a source manifest, semantic schema, profile, and issue register. Do not modify source data during this stage.

## 3. Split before learned preparation

Use `split-datasets`; add `stratified-sampling` or grouped/temporal logic as dictated by the prediction contract. Persist row or entity assignments and a random seed. Reserve the test set for one final evaluation.

## 4. Explore without contaminating evaluation

Use training data for transformation decisions. Run `target-analysis` first, then the smallest relevant set of univariate, distribution, bivariate, correlation, statistical, and multivariate skills. Validation/test summaries may be used only for consistency or drift checks, never to choose transformations or thresholds.

## 5. Define cleaning policies

Apply missing, duplicate, outlier, inconsistency, and invalid-data skills. Separate deterministic row-level rules from learned transforms. Fit learned cleaning parameters on training folds and report rows corrected, quarantined, or removed by reason.

## 6. Engineer and preprocess

Create typed features using the numerical, categorical, datetime, or text skills. Choose encoding, scaling, distribution transforms, selection, and reduction based on model family and diagnostics. Assemble all learned steps with `build-preprocessing-pipeline`; select CV with `design-cross-validation`; put imbalance treatment inside training folds.

## 7. Validate and release

Run leakage detection before and after feature engineering. Validate feature contracts and dataset consistency, establish drift baselines, then run model readiness. A release bundle should contain:

- source manifest and immutable input identifiers;
- target/prediction contract and exclusion list;
- persisted split assignments and seed;
- schema and data-quality reports;
- fitted preprocessing pipeline plus feature names;
- validation/readiness report with actual counts and metrics;
- version metadata for code, data, configuration, and artifacts.

## Failure and rerun policy

Fail closed on schema breaks, duplicate entities across splits, target leakage, non-finite model matrices, or unfitted/unsaved preprocessing. Quarantine correctable bad rows; never silently discard them. Rerun downstream stages when an upstream schema, target, split, cleaning rule, or feature definition changes. Do not overwrite versioned datasets or model artifacts.
