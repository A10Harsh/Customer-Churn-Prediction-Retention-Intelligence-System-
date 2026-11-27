# Phase 6A: development-only model optimization

Run `python -m retention.models.optimization --config params.yaml` from the
repository root. The command resumes four persistent Optuna studies in
`artifacts/model_optimization/optuna.db` and writes reports under
`reports/generated/model_optimization/`. It reads only the configured
`development.parquet`; it does not refit a final model, register a model, or
evaluate the held-out test set.

## Search and reproducibility

The configured budgets are 20 Logistic Regression, 20 Random Forest, 20
XGBoost, and 10 ANN trials. Each study maximizes mean five-fold Average
Precision (AP), with sequential trials and a seed 42 TPE sampler. The five
outer stratified folds are created by the shared Phase 4 fold factory. The
studies store and validate the development SHA-256, fold hash, seed, search
space version, feature mode, and starting model configuration before resuming.
Changing any of these requires a separate study rather than mixing results.

| Model | Features | Search space |
| --- | --- | --- |
| Logistic Regression | Engineered, linear preprocessing | C 0.001–100 log scale; unweighted or balanced class weights; L2 lbfgs, 2000 iterations |
| Random Forest | Engineered, tree preprocessing | 100–500 trees in steps of 100; depth 3, 5, 8, 12, 20 or unlimited; split 2–20; leaf 1–20; max features 0.3–1.0; unweighted or balanced |
| XGBoost | Baseline, tree preprocessing | 100–600 trees; depth 2–6; learning rate 0.01–0.2 log scale; minimum child weight 1–10; row and column sampling 0.7–1.0; L2 regularization 0.001–10 log scale; unweighted or training-fold scale positive weight |
| ANN | Baseline, neural preprocessing | Hidden widths (32,16), (64,32), (64,32,16); dropout 0–0.4; learning rate 0.0001–0.005 log scale; weight decay 0.000001–0.01 log scale; batch 32, 64 or 128; unweighted or weighted BCE |

Each fold fits fresh preprocessing on training rows only. Weight ratios use
the training labels of that fit. The ANN chooses epochs using an inner
stratified split, then refits a new network on the full outer training fold.
Outer validation labels never guide early stopping. The original configuration
is retained for a model family if its mean fold AP exceeds the best trial.

The selection criterion is mean five-fold AP across the four retained
candidates. When candidates are within 0.005 AP of the leader, the simpler
model is preferred where operationally suitable. This 0.005 margin is an
engineering heuristic, not evidence of statistical equivalence. Fold standard
deviations, pooled out-of-fold metrics, compute time, and interpretability
inform the decision.

Only the selected candidate undergoes calibration. In each outer fold, five
inner OOF predictions from outer training rows fit a sigmoid calibrator. A
fresh base model fitted to all outer training rows supplies outer validation
scores. Sigmoid is selected only if both pooled Brier score and log loss
improve over raw scores. The selected development OOF probabilities determine
an F1-maximizing threshold, with ties resolved by recall then higher threshold.
The default threshold 0.5 is retained as a reference. Threshold metrics are
selection-stage estimates from reused folds and may be optimistic.

Reports include trial history, best hyperparameters, fold and pooled OOF
metrics, before/after comparison, calibration scores, threshold table, a
frozen provisional configuration, and three essential figures. MLflow uses
the existing SQLite backend with parent model runs and child trial runs.
Generated reports and tracking files are local; no customer IDs or raw records
are logged.

## Measured results

All 70 trials completed: Logistic Regression 20 in 6.94 seconds, Random
Forest 20 in 81.58 seconds, XGBoost 20 in 14.79 seconds, and ANN 10 in
89.81 seconds of measured trial execution. One XGBoost attempt hit an OpenMP
runtime error before trial execution; the study then resumed with one worker
and completed its full budget. These timings exclude study setup, reruns for
final OOF reports, and calibration. They are hardware dependent.

| Model | Initial mean AP | Selected search mean AP ± fold SD | Pooled OOF AP | Pooled ROC-AUC | Five-fold fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6576 | 0.6580 ± 0.0445 | 0.6538 | 0.8470 | 0.24 |
| Random Forest | 0.6426 | 0.6608 ± 0.0372 | 0.6565 | 0.8459 | 2.87 |
| XGBoost | 0.6610 | 0.6629 ± 0.0459 | 0.6605 | 0.8485 | 0.56 |
| ANN | 0.6502 | 0.6527 ± 0.0463 | 0.6436 | 0.8418 | 6.90 |

XGBoost leads mean AP, but Logistic Regression falls within 0.005 (difference
0.00487) and is much simpler to fit and explain. The provisional selection is
engineered-feature, unweighted Logistic Regression with C=0.4338083764,
L2 regularization, lbfgs solver, and 2000 maximum iterations. The full best
trial settings for every family are in `best_hyperparameters.json`.

For the selected model, raw OOF probabilities scored Brier 0.134417, log loss
0.415135, AP 0.653798, and ROC-AUC 0.846955. Nested sigmoid calibration
scored Brier 0.135497, log loss 0.422065, AP 0.653721, and ROC-AUC
0.846855. Calibration therefore added complexity without improving either
calibration loss; raw probabilities remain selected.

At threshold 0.5, pooled OOF precision was 0.67597, recall 0.53445, F1
0.59694, specificity 0.90747, with 383 false positives and 696 false
negatives. The F1-maximizing threshold is **0.3389173724**. At that threshold,
precision was 0.56902, recall 0.72241, F1 0.63660, specificity 0.80237,
with 818 false positives and 415 false negatives. These are selection-stage
estimates, not independent validation.

Phase 6B should perform final training integration and a single held-out
evaluation after the provisional configuration is reviewed. Reusing the same
development folds for tuning and selection can inflate estimated performance.
Calibration and threshold choices are also based on development data. A
resumed TPE study validates compatibility but can take a different trial path
from an uninterrupted run because sampler random state is not persisted.
No intervention costs are verified, so maximum F1 is the threshold objective
and no financial return is claimed.
