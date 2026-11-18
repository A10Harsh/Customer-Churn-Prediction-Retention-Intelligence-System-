---
name: handle-class-imbalance
description: Choose metrics, class weights, fold-local resampling, calibration, and thresholding for imbalanced classification without contaminating validation or test data.
---

# Handle Class Imbalance

## Purpose and use cases

Train and evaluate classifiers when the positive class is rare or costs are asymmetric.

## Inputs and outputs

Inputs: training target, feature types, CV splitter, estimator, business costs/capacity, seed. Outputs: imbalance strategy/pipeline, fold metrics, calibrated probabilities if needed, and validation-selected threshold recommendation.

## Execution and decision logic

1. Establish baseline prevalence and report PR-AUC, ROC-AUC, precision, recall, and confusion counts; accuracy is insufficient.
2. Start with class weights/cost-sensitive learning when supported.
3. If sampling helps, place RandomOverSampler/SMOTE/under-sampling inside an imbalanced-learn pipeline and inside each training fold only; prefer simple oversampling for mixed/sparse features unless a valid distance exists.
4. Keep validation/test at natural prevalence. Calibrate on held-out or nested folds.
5. Select operating threshold on validation using declared cost/capacity; evaluate once on test and do not promote/change threshold without approval.

## Python example

```python
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def imbalance_pipeline(preprocess: object, seed: int = 42) -> Pipeline:
    """Build a fold-safe oversampling classifier pipeline."""
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("sample", RandomOverSampler(random_state=seed)),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
```

## Validation checks and best practices

Assert samplers never touch validation/test, fold prevalence is reported, probabilities are calibrated when used operationally, and all seeds are fixed. Compare against an unsampled baseline and report real fold/test metrics only.
