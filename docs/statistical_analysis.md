# Automated statistical-analysis pipeline

Phase 3B Step 12 promotes only stable, operationally useful methods from
Notebooks 08–15 into reusable production modules. The notebooks remain the full
research and educational record; simulations, extensive diagnostics, multiple-group
post-hoc comparisons, and interview material are intentionally not automated.

## Components

- `inference.py`: Wilson proportion intervals, Welch mean-difference inference,
  Mann–Whitney U, Cohen's d, Cliff's delta, Pearson chi-square expected-frequency
  checks, justified Fisher fallback, and corrected Cramér's V reuse.
- `multiplicity.py`: family-specific Benjamini–Hochberg FDR and Holm FWER
  adjustment, including a documented positive floating-point bound for underflowed
  p-values.
- `multicollinearity.py`: auxiliary-regression VIF/tolerance, temporary
  reference-category encoding, standardized singular values, rank, and condition
  diagnostics.
- `statistical_analysis.py`: configuration, development-only safeguards,
  orchestration, deterministic artifacts, atomic publication, and CLI.
- `reporting.py`: shared strict JSON and deterministic CSV utilities plus the
  statistical HTML report.

## Methods and hypothesis families

The primary `primary_feature_churn` family contains 13 registered tests: three
Welch tests for tenure, MonthlyCharges, and TotalCharges, plus ten Pearson
chi-square tests for service, billing, and demographic predictors. The separate
`complementary_numerical_ranks` family contains the three Mann–Whitney analyses.
Benjamini–Hochberg at configured q=0.05 is the exploratory primary procedure;
Holm at alpha=0.05 is the conservative sensitivity procedure.

Numerical direction is always churned minus retained. Pearson chi-square uses no
continuity correction, matching the prior notebooks and corrected Cramér's V.
Fisher's exact test is run only for an inadequately supported 2×2 table. Wilson
intervals cover the overall churn proportion and configured Contract segments;
Welch–Satterthwaite intervals cover numerical mean differences.

VIF is calculated from an auxiliary regression with an intercept:

`VIF_j = 1 / (1 - R²_j)` and `Tolerance_j = 1 / VIF_j`.

The mixed diagnostic design uses the EDA categorical feature list, treats
SeniorCitizen as categorical, drops one reference level per feature, and is never
persisted. Nonconstant columns are standardized only for the condition number.
Infinite VIF and a null condition-number value in JSON are accompanied by explicit
rank-deficiency status strings.

## Outputs

Run:

```bash
python -m retention.analysis.statistical_analysis
```

or reproduce through DVC:

```bash
dvc repro statistical_analysis
dvc status
dvc dag
```

The atomically published directory `reports/generated/statistical_analysis/`
contains:

- `summary.json`: strict, sorted, timestamp-free machine summary.
- `numerical_hypothesis_tests.csv`: Welch, Mann–Whitney, confidence intervals,
  and numerical effect sizes.
- `categorical_hypothesis_tests.csv`: chi-square support diagnostics, optional
  Fisher results, and corrected Cramér's V.
- `adjusted_hypothesis_tests.csv`: raw, BH, and Holm values and decisions within
  explicit families.
- `confidence_intervals.csv`: overall, numerical-difference, and configured
  segment intervals.
- `multicollinearity.csv`: numerical and reference-coded mixed VIF diagnostics.
- `report.html`: concise evidence, uncertainty, dependency, and recommendation
  summary with relative artifact links.

## Reconciled findings

Production results match Notebooks 09, 11, 12, 14, and 15 within floating-point
tolerance. The development churn rate is 0.26535 with a 95% Wilson interval of
0.25399–0.27704. Numerical Cohen's d values are -0.85146 for tenure, +0.43557 for
MonthlyCharges, and -0.46467 for TotalCharges. Contract has corrected Cramér's
V=0.41208.

The primary family rejects 12 of 13 null hypotheses under raw, BH, and Holm
decisions; gender remains the only non-rejection, so correction changes no primary
decision. All three complementary rank tests remain rejected.

Numerical-only VIF is 5.8835 for tenure, 3.2045 for MonthlyCharges, and 9.5813 for
TotalCharges. The temporary mixed design has rank 24 across 31 columns including
the intercept, a deficiency of seven. Nine encoded columns participate in exact
phone/internet service identities and therefore have infinite VIF. These results
diagnose coefficient identifiability under this coding; they do not mandate feature
removal.

## Limitations and Phase 3C recommendations

Questions were informed by EDA on the same development records, so adjusted
significance is not independent confirmation. Observational associations do not
establish causation. Dummy-column VIF depends on reference coding, and the IBM data
have no documented probability-sampling or temporal design.

For Phase 3C, compare explicit availability-plus-adoption service encodings, use
regularization where coefficient stability matters, and evaluate tenure/charge
redundancy through leakage-safe cross-validation. Tree models may retain correlated
predictors. No feature should be removed from a marginal p-value or VIF alone.

The DVC stage depends on `development.parquet`, production source modules,
`requirements-lock.txt`, and relevant parameters. It does not depend on notebooks,
`test.parquet`, or complete validated data, and it does not regenerate the split.
