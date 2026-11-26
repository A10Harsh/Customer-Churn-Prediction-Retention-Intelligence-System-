# Classical model training and local experiments

Phase 3D and Phase 4 use only `data/processed/development.parquet`. Run the
configured 13 experiments with:

```bash
python -m retention.models.training --config params.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

The second command opens the local MLflow UI at `http://127.0.0.1:5000`.
The tracking backend is SQLite (`mlflow.db`) and artifacts are local in
`mlruns/`. These, `mlflow.db`, and `reports/generated/model_training/` are excluded from Git.
The CLI does not read the held-out test set or save production models.

## Architecture and initial settings

`split_features_target` drops `customerID` and maps Churn `No=0`, `Yes=1`.
One shuffled five-fold `StratifiedKFold` with seed 42 is generated once and
shared by every configuration. Each fold creates a fresh scikit-learn Pipeline
containing the baseline or engineered preprocessor and one estimator. All
imputation, scaling, category mapping, and model fitting occur on training rows
only. Out-of-fold positive-class probabilities are placed back in development
row order; every row is predicted once. No resampling or global preprocessing
is used.

| Model | Initial settings | Imbalance comparison |
| --- | --- | --- |
| DummyClassifier | `strategy="prior"` | One unweighted baseline. |
| Logistic Regression | L2 default, `C=1`, `max_iter=1000` | None versus `class_weight="balanced"`. |
| Random Forest | 100 trees, minimum leaf 2, square-root feature sampling, `n_jobs=2` | None versus `class_weight="balanced"`. |
| XGBoost | 150 trees, depth 3, learning rate 0.05, row/column subsampling 0.9, CPU histogram, `n_jobs=2` | `scale_pos_weight=1` versus training-fold negatives / positives. |

Each non-dummy model uses both the 19-predictor baseline and 25-predictor
engineered representation, with unweighted and weighted configurations. These
are initial fixed settings, not tuned hyperparameters. Optimization, thresholds,
calibration, ANN training, and held-out evaluation are later work.

## Metrics and logged artifacts

Average Precision (AP) is primary and is computed from probabilities. AP is
the stepwise precision weighted by increases in recall; it is not trapezoidal
PR-AUC. ROC-AUC also uses probabilities. Precision, recall, F1, balanced
accuracy, accuracy, specificity, and confusion counts use threshold 0.5.
Training and inference seconds are measured per fold. Reports separate fold
values, fold means and sample standard deviations, and pooled out-of-fold
metrics. Zero predicted positives give precision and F1 of zero.

One MLflow run per configuration logs hyperparameters, feature mode, imbalance
strategy, fold count, seed, threshold, development SHA-256, Git commit and dirty
status, fold/CV/pooled metrics, and evaluation artifacts. Per-run artifacts are
`configuration.json`, `fold_metrics.csv`, and `oof_evaluation.csv` (fold number,
binary outcome, probability; no customer identifiers or predictors). The
report directory contains `leaderboard.csv`, combined `fold_metrics.csv`,
`summary.json`, AP/ROC comparison, leading PR curves, and leading confusion
matrices. Local tracking files and generated reports remain untracked.

## Initial development results

All 13 configurations completed on 5,634 development customers. The dummy
mean AP was 0.2654. The following are **mean five-fold AP**, not final test
results; full metrics and run IDs are in `leaderboard.csv`.

| Model | Features | Unweighted AP | Weighted AP |
| --- | --- | ---: | ---: |
| Logistic Regression | Baseline | 0.6520 | 0.6505 |
| Logistic Regression | Engineered | 0.6576 | 0.6562 |
| Random Forest | Baseline | 0.6369 | 0.6335 |
| Random Forest | Engineered | 0.6426 | 0.6409 |
| XGBoost | Baseline | 0.6599 | **0.6610** |
| XGBoost | Engineered | 0.6604 | 0.6592 |

The top XGBoost configuration had mean ROC-AUC 0.8479, precision 0.5266,
recall 0.8054, and mean fold fit time 0.078 s. Engineered features increased
mean AP by roughly 0.0056 for Logistic Regression and Random Forest under
unweighted fitting, but changed XGBoost AP by only 0.0004. Weighting strongly
raised recall and lowered precision at threshold 0.5; AP changes were small
and mixed. Paired folds show engineered-minus-baseline AP differences of
0.0056, 0.0057, and 0.0004 for unweighted Logistic Regression, Random Forest,
and XGBoost respectively. Weighted-minus-unweighted baseline AP differences
were -0.0015, -0.0033, and +0.0011. The strongest initial configuration for
each model is engineered unweighted Logistic Regression (0.6576), engineered
unweighted Random Forest (0.6426), and baseline weighted XGBoost (0.6610).

The XGBoost and Logistic Regression results are close. Training cost is higher
for Random Forest than XGBoost in this initial local run; these small timing
measurements depend on hardware and workload. XGBoost, plus the competitive
Logistic Regression baseline, are sensible candidates for later Optuna work.
No production model is selected. Repeated comparisons on the same CV folds can
make the chosen configuration's apparent performance optimistic. The untouched
held-out dataset remains reserved for final evaluation.

## Interview explanations

- **Why stratified cross-validation?** It maintains the observed churn fraction
  in each fold, making minority-class metrics more stable while allowing every
  development row one validation prediction.
- **Why AP instead of accuracy?** Churn is a minority outcome; AP evaluates the
  ranking of positive cases across recall levels, whereas predicting everyone
  as retained can still achieve high accuracy.
- **How does class weighting help?** It increases the loss contribution of
  churners during fitting. It may improve recall, but can lower precision and
  does not guarantee better ranking.
- **How does MLflow improve reproducibility?** Each run records configuration,
  data checksum, source revision, fold metrics, and artifacts so a comparison
  can be traced to its inputs and rerun.
