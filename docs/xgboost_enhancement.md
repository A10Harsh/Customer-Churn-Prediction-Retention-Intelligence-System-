# Phase 6A+: targeted XGBoost enhancement

Run `python -m retention.models.xgboost_enhancement --config params.yaml` from
the repository root. This development-only command resumes a separate
50-trial Optuna study and writes reports to
`reports/generated/xgboost_enhancement/`. It reads the 5,634-row
`development.parquet` and existing Phase 6A OOF reports. It neither reads
held-out observations nor fits a final model.

## Design and leakage controls

The original seed 42 and five outer stratified fold assignments are verified
against the Phase 6A fold hash. Each fold creates fresh preprocessing and
estimator components. Charge-band quartiles in the enhanced feature set are
learned only on fold training rows. The six proposed additions are contract ×
tenure group, payment × tenure group, internet type × charge band, support and
security status, monthly charge per service, and tenure at most six months.
Zero-service charge ratios become missing and enter the existing imputation
path; unseen categories enter the existing one-hot unknown-category path.

SMOTE-NC was applied only to raw training-fold predictors before encoding.
Neighbor distances used training-fitted numerical scaling; synthetic tenure
was rounded to whole months. Every synthetic row was checked against the
existing phone and internet consistency constraints, finite nonnegative
charges, and zero-tenure charge consistency. Invalid synthetic rows are
rejected. Dependent engineered features are recomputed after sampling.
Model-based feature selection used a training-only XGBoost importance model
inside the fitted pipeline. No validation rows were resampled or selected on.

The search maximized mean five-fold Average Precision (AP) with a seeded
TPESampler, sequential trials, CPU histogram training, and no early stopping
on outer validation rows. The search ranged over 200–1000 trees, depth 1–5,
learning rate 0.005–0.1, minimum child weight 1–15, row sampling 0.65–1.0,
column sampling 0.6–1.0, L2 regularization 0.0001–100, optional L1
regularization 0.0001–10, gamma 0–5, and positive weight 1–3.5. The previous
configuration was enqueued as a compatible reference trial. Study metadata
validates data checksum, folds, seed, feature and sampling configuration,
search version, and previous candidate before resuming.

## Screening results

All values below are mean five-fold AP. The 0.003 minimum gain for retaining
extra feature or resampling complexity was set before screening and is an
engineering heuristic.

| Feature configuration | Mean AP |
| --- | ---: |
| Baseline | 0.66291 |
| Existing engineered | 0.66252 |
| Proposed enhanced | 0.66377 |

The enhanced feature set gained only 0.00086 AP over baseline, so baseline
features were retained for the extended search.

| Training strategy on baseline features | Mean AP | Effective positive/negative ratio | Synthetic rows accepted / rejected across folds |
| --- | ---: | ---: | ---: |
| Unweighted | 0.66291 | Original ratio | 0 / 0 |
| Fold-specific class weight | 0.66314 | Original ratio | 0 / 0 |
| SMOTE-NC target 0.5 | 0.66394 | 0.49988 | 2,296 / 0 |
| SMOTE-NC target 0.75 | 0.66028 | 0.74994 | 6,436 / 0 |
| SMOTE-NC target 1.0 | 0.65628 | 1.00000 | 10,576 / 0 |

The SMOTE-NC 0.5 gain was 0.00104 AP, below the retention rule; oversampling
was excluded. The controlled reduced-feature pipeline scored 0.66206 AP,
also below the complete baseline feature set.

## Extended search and calibration

All 50 trials completed in 44.91 seconds of measured objective execution.
The best was trial 35, with mean fold AP **0.66637 ± 0.04334** and pooled raw
OOF AP **0.66476**. This is +0.00346 mean AP over the previous optimized
XGBoost, +0.00533 over initial weighted XGBoost, and +0.00833 over optimized
Logistic Regression. These are paired development-fold comparisons, not an
independent test. Full fold differences are in `paired_fold_differences.csv`.

The best model uses baseline features, 535 trees, depth 1, learning rate
0.03579038, minimum child weight 10, subsample 0.70521663,
column sample 0.95949467, L2 regularization 20.77581123, L1 regularization
0.18999892, gamma 2.88655385, and positive weight 2.99053677. It uses
CPU histogram training with one worker. No oversampling or feature reduction
is retained.

The nested outer-fold calibration procedure fit each sigmoid on inner OOF
predictions from outer training rows only. Raw versus sigmoid results were:

| Probability strategy | Brier | Log loss | Pooled AP | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Raw | 0.16687 | 0.49525 | 0.66476 | 0.84944 |
| Sigmoid | **0.13436** | **0.41518** | 0.66301 | 0.84928 |

Sigmoid is retained because both calibration losses improved substantially.
AP and ROC-AUC were slightly lower after calibration.

## Operating point and revised selection

The same maximum-F1 and maximum-recall-at-55%-precision policies were applied
to all six compared candidates. The complete results, including confusion
counts and predicted-positive proportions, are in `threshold_comparison.csv`.
For the calibrated enhanced XGBoost candidate:

| Policy | Threshold | Precision | Recall | F1 | TP | FP | FN | Predicted positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 0.50000 | 0.64615 | 0.58997 | 0.61678 | 882 | 483 | 613 | 24.23% |
| Maximum F1 | 0.38274 | 0.58827 | 0.71104 | 0.64385 | 1,063 | 744 | 432 | 32.07% |
| Maximum recall with precision ≥0.55 | **0.33045** | **0.55061** | **0.75318** | 0.63616 | **1,126** | **919** | **369** | **36.30%** |

The constrained threshold identifies 244 more churners than 0.5 and produces
436 more false alerts. Its recall is below the proposed 80–85% target. All
six candidates can meet the 0.55 precision floor at some threshold; none
reaches 0.80 recall under that floor. Threshold changes do not improve AP or
ROC-AUC. Retention workload rises from 1,365 predicted positives at 0.5 to
2,045 at the constrained threshold; no intervention cost or business return
is assumed.

The revised provisional candidate is calibrated XGBoost with threshold
**0.3304499867**, recorded separately in `revised_selection.json`. Mean
five-fold AP remains the primary selection criterion. The previous 0.005
AP simplicity tolerance is preserved. The previous optimized XGBoost is
within 0.005 AP and has slightly higher constrained recall (0.7565 versus
0.7532), but both have the same operational model family; the higher-AP,
better-calibrated extended configuration is selected. Logistic Regression
and the other optimized families are more than 0.005 AP behind the extended
candidate. The original Phase 6A selection remains untouched.

## Limits and Phase 6B

The same development folds supported screening, 50-trial optimization,
calibration comparison, threshold choice, and model selection. The reported
values are selection-stage estimates and may be optimistic. The proposed AP
0.68, ROC-AUC 0.85, recall 0.80–0.85 at precision at least 0.55, and F1
0.65 targets were not achieved together. Phase 6B must freeze the reviewed
configuration, integrate final training, and perform one independent held-out
evaluation. No final refit, model registration, or DVC training integration
was performed here.
