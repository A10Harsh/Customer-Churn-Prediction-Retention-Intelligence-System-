# PyTorch ANN development experiments

Run the four ANN configurations on the configured development split:

```bash
python -m retention.models.ann_training --config params.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

The local MLflow experiment is `customer-retention-ann`. The 13 earlier
classical-model runs stay in `customer-retention-classical-ml`. ANN reports are
written under `reports/generated/ann_training/`; classical reports are read
for comparison and are not retrained or overwritten.

## Network and training

`RetentionANN` is a feedforward network with input width inferred after fitted
neural preprocessing, hidden widths 64, 32, and 16, ReLU activations, dropout
0.2, and one raw output logit. It has 4,609 parameters for baseline's 30
encoded inputs and 4,737 for engineered features' 32 encoded inputs.
The fixed initial optimizer is AdamW (learning rate 0.001, weight decay
0.0001), batch size 64, at most 100 epochs, and patience 10. Training uses
float32 tensors, `BCEWithLogitsLoss`, and CPU by default. MPS can be requested
when available; it was unavailable in this run. The threshold is 0.5.

The same five stratified outer folds and seed 42 as the classical experiments
are reconstructed by the shared fold factory. Within each outer training fold,
a reproducible stratified 20% inner validation split chooses the epoch with
best Average Precision. Preprocessing is fitted only on the inner training
part. A fresh preprocessor and network then refit on the **complete outer
training fold** for the selected epoch count. Outer validation labels do not
guide early stopping. Weighted loss uses negatives / positives independently
for the inner fit and the outer refit. Every development row receives exactly
one out-of-fold probability in original row order. Fold-specific networks are
not saved as production artifacts.

Python, NumPy, and PyTorch seeds are set for each fit, with one CPU thread for
repeatability. Exact values can still vary with hardware, PyTorch version,
device kernels, and library versions, especially when opting into MPS.

## Actual development results

The four configurations completed on 5,634 development customers. Values are
mean five-fold metrics; standard deviation for AP is shown separately. The
complete [ANN leaderboard](../reports/generated/ann_training/leaderboard.csv)
contains run IDs and further metrics.

| Features | Loss | AP ± SD | ROC-AUC | Precision | Recall | F1 | Mean best epoch | Mean fold training seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | Weighted | **0.6502 ± 0.0482** | 0.8410 | 0.5058 | 0.7940 | 0.6178 | 8.0 | 0.735 |
| Engineered | Weighted | 0.6469 ± 0.0461 | 0.8411 | 0.5089 | 0.8027 | 0.6203 | 9.0 | 0.814 |
| Baseline | Unweighted | 0.6468 ± 0.0538 | 0.8418 | 0.6632 | 0.5298 | 0.5880 | 17.8 | 1.355 |
| Engineered | Unweighted | 0.6446 ± 0.0443 | 0.8436 | 0.6566 | 0.5632 | 0.6042 | 11.0 | 0.911 |

Paired fold mean AP differences for weighted minus unweighted were +0.0034
with baseline features and +0.0023 with engineered features. Engineered minus
baseline differences were -0.0034 under weighted loss and -0.0022 without it.
Weighting raised recall and lowered precision at threshold 0.5. The best ANN
mean AP (0.6502) was below the existing best Logistic Regression (0.6576)
and XGBoost (0.6610), but above Random Forest (0.6426). The 20 ANN folds took
19.07 seconds of measured selection-plus-refit time in this CPU run, compared
with 6.72 seconds of fold fit time across all 65 earlier classical-model folds.
These costs are hardware-dependent and the timing scopes differ slightly.

Per-run `fold_metrics.csv`, `training_history.csv`, `oof_evaluation.csv`, and
`configuration.json` record fold weights, selected epochs, metrics, checksum,
device, and seed. The report directory also contains a consolidated classical
and ANN leaderboard, a representative training-history figure, and one
precision–recall comparison. MLflow has one principal run per ANN
configuration. No raw predictors, customer IDs, credentials, or models are
logged. Generated reports and local MLflow files are excluded from Git.

The weighted baseline ANN is an initial candidate for later Optuna experiments,
alongside competitive classical models. Repeated comparisons on the same CV
folds can make the selected configuration's estimate optimistic. No production
model, threshold, or calibration is chosen here; the held-out test split remains
untouched for final evaluation.

## Interview notes

- **BCEWithLogitsLoss:** It combines sigmoid and binary cross-entropy in a
  numerically stable loss. The network emits logits during training; sigmoid
  is applied only when probabilities are needed.
- **Dropout:** It randomly disables hidden activations during training to
  discourage dependence on a few units. Eval mode disables dropout.
- **Early stopping:** Inner validation AP selects an epoch count, then a new
  network refits on all outer training rows for that count. The outer fold is
  reserved for evaluation.
- **AdamW:** It adapts parameter updates while applying decoupled weight
  decay. Its learning rate and decay are fixed here, not tuned.
