# Model card: customer churn predictor candidate

**Status:** Evaluated candidate, not production or champion. **Intended use:**
prioritize analyst review of possible churn among customers with the same
raw predictor schema as the IBM Telco Customer Churn dataset. The output is
a calibrated churn probability and a classification at the frozen threshold
`0.3304499866713542`. It is not a causal estimate of retention benefit.

**Training:** 5,634 development customers, baseline features, XGBoost with
fixed hyperparameters and class weight, followed by a sigmoid fit on raw
five-fold out-of-fold scores. The held-out 1,409 rows were excluded from all
training and calibration.

**Held-out performance:** AP 0.6729, ROC-AUC 0.8502, precision 0.5551,
recall 0.7406 and F1 0.6346. The precision 95% paired-row bootstrap
interval is 0.5108–0.5965. The 55% precision target was met on this sample;
its lower uncertainty bound was below 55%. See [full evaluation](final_model.md)
for the confusion matrix, calibration metrics and reference threshold.

**Limitations:** The model misses many churners with long tenure, one- or
two-year contracts, no internet service, or dependents. Sparse groups have
imprecise or undefined metrics. The dataset is a static random split, so
these results do not prove future temporal generalization. Retention impact
and financial benefit have not been measured.

**Monitoring:** Track schema and category drift, calibration, AP, precision,
recall, false-negative rate and false-positive rate overall and by the
predefined diagnostic segments. Require new independently collected data
before changing the threshold or promoting this candidate.
