# Phase 6B final model

The frozen candidate is a baseline-feature XGBoost model with a five-fold
development-OOF sigmoid calibrator. The decision threshold is
`0.3304499866713542`; a probability at or above it predicts churn. The
preprocessing pipeline, estimator, calibrator, threshold, schema and
fingerprints are stored together in
`artifacts/final_model/predictor.joblib`. Load this joblib file only from a
trusted source.

The final estimator was fitted on all 5,634 development customers. The
calibrator was fitted separately on row-aligned, raw out-of-fold probabilities
from five stratified development folds; no in-sample scores or held-out rows
were used. Its fitted coefficient is `5.288934911791401` and intercept is
`-3.795024226744771`. The frozen selection SHA-256 is
`445ea3103e21c0c51be006ff3d1263435c78eb245034ebd1fec890bf9bdd77bb`.
The development SHA-256 is
`28d12b92122ab94a8172be83e2bcf7e93b128a1bf3633c2b48075c426864da13`.

## Independent evaluation

The first held-out evaluation used 1,409 customers once, with the fitted
artifact and frozen threshold. Its immutable local record is
`artifacts/final_model/heldout_evaluation.json`; the report is in
`reports/generated/final_evaluation/`. The precision floor of 0.55 was met
on this split, but the 95% interval includes values below 0.55.

| Metric | Development pooled OOF | Held-out | Held-out 95% bootstrap interval |
| --- | ---: | ---: | ---: |
| Calibrated AP | 0.6630 | 0.6729 | 0.6242–0.7201 |
| Calibrated ROC-AUC | 0.8493 | 0.8502 | 0.8282–0.8704 |
| Precision | 0.5506 | 0.5551 | 0.5108–0.5965 |
| Recall | 0.7532 | 0.7406 | 0.6931–0.7844 |
| F1 | 0.6362 | 0.6346 | 0.5949–0.6691 |

At the frozen threshold, accuracy was 0.7736, balanced accuracy 0.7631,
specificity 0.7855, Brier score 0.1348 and log loss 0.4142. There were
277 true positives, 222 false positives, 813 true negatives and 97 false
negatives. The model predicted positive for 35.42% of customers. Batch
inference on the test split took 0.030 seconds in this local run.

Held-out raw-score AP was 0.6729 and ROC-AUC was 0.8502. The fitted sigmoid
is monotonic, so calibrated-score AP and ROC-AUC were the same; calibration
changed probability values and fixed-threshold decisions, not ranking.

At the predefined 0.5 reference threshold, precision was 0.6404, recall
0.5856 and F1 0.6117; there were 219 true positives, 123 false positives,
912 true negatives and 155 false negatives. This reference did not alter the
frozen operating point.

Intervals are 2,000 paired-row percentile bootstrap resamples of fixed
predictions and labels, seeded at 42. All 2,000 were valid. They measure
test-sample uncertainty conditional on the fitted model and do not include
training, tuning or future drift uncertainty.

## Segment limitations and monitoring

The month-to-month contract group had recall 0.842 and FPR 0.483. One-year
and two-year contracts had zero observed recall, with few positive cases;
two-year precision was undefined because the model issued no positive
predictions. Customers with tenure 49–72 months had recall 0.267 and FNR
0.733. DSL recall was 0.604, and customers without internet service had
recall 0.348. Dependents had recall 0.434 versus 0.791 for customers without
dependents. These are diagnostic slices, not grounds to change the model or
threshold using the test set. Segment error-rate Wilson intervals and sparse
flags appear in `evaluation.json`.

Monitor input schema/category drift, score distribution, calibration,
precision, recall, FNR and FPR by contract, tenure, internet service and
demographic groups. Review sparse segments before drawing conclusions.
This static random split does not establish future temporal generalization
or retention intervention effectiveness.

## Commands

Run from the repository root with the project virtual environment active:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export DVC_SITE_CACHE_DIR=/tmp/churn-dvc-site-cache
dvc repro final_train
dvc repro final_evaluate
dvc status
python -m retention.models.final_registry
```

The registry command is explicit and separate from DVC caching. It creates
or reuses `customer-churn-predictor` version 1 with alias `candidate`; it
does not promote to production. The verified MLflow run is
`6d85e1a34e804b9a9954b1d03e692c9f` in experiment
`customer-retention-final-model`.

For local single or batch inference, load the trusted artifact with
`retention.models.predictor.load_predictor` and call `predict` with a
dictionary of one customer's original raw predictors or a DataFrame of
customers. The result has `churn_probability`, `predicted_churn` and
`threshold`; `raw_scores` exposes uncalibrated scores for diagnostics.
Neither customer ID nor the target belongs in prediction input.
