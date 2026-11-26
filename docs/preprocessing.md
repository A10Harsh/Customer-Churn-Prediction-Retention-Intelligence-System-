# Feature engineering and preprocessing

Phase 3C supplies deterministic candidate features and **unfitted** preprocessing
factories. Churn modelling, feature-subset experiments, and fitted artifact
publication are planned for a later phase.

## Candidate features

`TelcoFeatureEngineer` uses the configured lifecycle boundaries from `params.yaml`.
All calculations are customer-local and exclude `customerID` and `Churn`.

| Feature | Definition |
| --- | --- |
| `tenure_group` | 0–6 onboarding, 7–12 first year, 13–24 developing, 25–48 established, 49–72 mature months. |
| `total_services` | Count one each for phone service, multiple lines, internet service, and each of six subscribed internet add-ons; range 0–9. Unavailable or unsubscribed services count zero. |
| `support_security_count` | Count subscribed TechSupport and OnlineSecurity; range 0–2. |
| `is_auto_payment` | 1 for automatic bank transfer or credit card; otherwise 0. |
| `is_month_to_month` | 1 for a month-to-month contract; otherwise 0. |
| `avg_monthly_spend` | `TotalCharges / tenure`; missing for zero or missing tenure, then imputed inside the training fold. |

The engineered representation retains `PhoneService` and `InternetService` for
service availability and internet type. It replaces `MultipleLines` and six
internet add-on status columns with adoption indicators. Thus an unavailable
service and an available but unsubscribed service both have adoption 0, while
the retained availability column distinguishes them. Inconsistent service
states and missing or invalid structural, contract, or payment categories raise
an error. Missing numeric values propagate to the fold-fitted imputer. The input
DataFrame is unchanged and remains available as the 19-predictor baseline.

These features may overlap with their source predictors. In particular,
`total_services` summarizes service indicators, `support_security_count` sums
two of them, `is_month_to_month` is determined by `Contract`, and
`avg_monthly_spend` combines `TotalCharges` and `tenure`. This is expected for
candidate comparisons and is not a reason for pre-validation elimination.

## Preprocessing configurations

`make_preprocessor(feature_set, model_family, selected_features=None)` returns
a new scikit-learn `Pipeline`. `feature_set` is `baseline` or `engineered`;
`model_family` is `linear`, `neural`, or `tree`. The baseline has 19 raw
predictors; the engineered representation has 25 before encoding.

| Component | Linear and neural | Tree |
| --- | --- | --- |
| Numerical | Median imputation, then `StandardScaler` | Median imputation |
| Binary | Most-frequent imputation, no scaling | Same |
| Categorical | Most-frequent imputation, one-hot encoding with first category dropped | Most-frequent imputation, full one-hot encoding |

`SeniorCitizen` is an unscaled binary indicator. The output is a dense numeric
matrix with stable, prefixed names from `get_feature_names_out()`, suitable for
scikit-learn estimators, XGBoost, and later PyTorch conversion. Encoded width
depends on categories observed in each training fold. Missing values use
statistics learned in that fold. Previously unseen categories are encoded as
all zeros by `OneHotEncoder(handle_unknown="ignore")`; in linear and neural
configurations that is also the dropped reference category's encoding. This
limits interpretation of unseen values. The engineered path rejects unknown
structural service, contract, and payment categories before encoding; other
unseen categorical values use the encoder rule.

`selected_features` accepts an explicit list of pre-encoding predictor names.
The factory rejects unknown or duplicate names and otherwise keeps all features.
It does not search for or select a subset automatically.

## Leakage-safe usage

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from retention.features.preprocessing import make_preprocessor, split_features_target

development = pd.read_parquet("data/processed/development.parquet")
X, y = split_features_target(development)  # drops customerID; No=0, Yes=1

pipeline = Pipeline(
    [
        ("preprocess", make_preprocessor("engineered", "linear")),
        ("estimator", LogisticRegression(max_iter=1000)),
    ]
)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Planned model-training phase: cross_validate(pipeline, X, y, cv=folds)
```

The example constructs a future estimator pipeline; it does not fit one. During
model evaluation, use the **same stratified folds** for baseline and engineered
comparisons. `cross_validate` clones the entire pipeline and fits imputation,
scaling, and category mappings only on each training fold. Never preprocess the
complete development set before cross-validation or create a global processed
dataset for it. The held-out test split remains reserved for final evaluation.

## Feature selection strategy and limits

Compare baseline and engineered configurations by cross-validated predictive
performance, not marginal p-values alone. Investigate numerical redundancy
through controlled `selected_features` comparisons when justified. Later,
regularized Logistic Regression can manage correlated predictors and model-based
importance can assess contributions. Neither a high VIF nor a correlation is
grounds for deleting a candidate now. No automated selector, balancing, churn
model, or fitted preprocessing artifact is provided in Phase 3C.
