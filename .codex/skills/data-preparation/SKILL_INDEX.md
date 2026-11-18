# Skill Index

| Stage | Skill | Use when |
|---|---|---|
| Understand | `ingest-data` | Load files, tables, or frames with provenance and safe parsing |
| Understand | `profile-data` | Produce a bounded structural and descriptive profile |
| Understand | `detect-schema` | Infer and persist column contracts and semantic roles |
| Understand | `infer-data-types` | Resolve physical versus semantic data types |
| Understand | `assess-data-quality` | Quantify completeness, uniqueness, validity, and integrity |
| EDA | `univariate-analysis` | Inspect each feature independently |
| EDA | `bivariate-analysis` | Analyze feature-pair or feature-target relationships |
| EDA | `multivariate-analysis` | Find interactions, clusters, and conditional structure |
| EDA | `statistical-analysis` | Estimate uncertainty and test pre-specified hypotheses |
| EDA | `correlation-analysis` | Measure suitable numeric and categorical associations |
| EDA | `distribution-analysis` | Diagnose skew, tails, zero inflation, and shift |
| EDA | `target-analysis` | Audit target validity, prevalence, and temporal behavior |
| Clean | `treat-missing-values` | Choose deletion, indicator, simple, or learned imputation |
| Clean | `remove-duplicates` | Detect exact, key, and conflicting duplicate records |
| Clean | `detect-handle-outliers` | Flag and treat implausible or influential extremes |
| Clean | `normalize-inconsistent-data` | Canonicalize representations without losing meaning |
| Clean | `correct-invalid-data` | Apply auditable domain rules and quarantine failures |
| Engineer | `engineer-numerical-features` | Build ratios, differences, bins, and interactions |
| Engineer | `engineer-categorical-features` | Consolidate levels and create domain groupings |
| Engineer | `engineer-datetime-features` | Create calendar, age, recency, and cyclic features |
| Engineer | `engineer-text-features` | Build leakage-safe lexical or vector features |
| Engineer | `encode-categories` | Select one-hot, ordinal, frequency, hashing, or target encoding |
| Engineer | `scale-features` | Choose standard, robust, min-max, or no scaling |
| Engineer | `transform-distributions` | Apply log, power, quantile, or clipping transforms |
| Engineer | `select-features` | Remove unsafe/redundant features and select within folds |
| Engineer | `reduce-dimensionality` | Use PCA/SVD only when compression benefits justify opacity |
| Preprocess | `split-datasets` | Create leakage-safe train/validation/test partitions |
| Preprocess | `design-cross-validation` | Match CV to iid, grouped, temporal, or imbalanced data |
| Preprocess | `stratified-sampling` | Preserve target or joint subgroup proportions |
| Preprocess | `handle-class-imbalance` | Select weights, sampling, metrics, and threshold workflow |
| Preprocess | `build-preprocessing-pipeline` | Assemble typed, serializable train/serve transformations |
| Validate | `detect-data-leakage` | Audit temporal, target, group, preprocessing, and proxy leakage |
| Validate | `validate-features` | Enforce feature names, types, ranges, nulls, and categories |
| Validate | `check-dataset-consistency` | Compare split and serving contracts and row identities |
| Validate | `detect-data-drift` | Compare reference and current feature distributions |
| Validate | `assess-model-readiness` | Run the final evidence-based release gate |
