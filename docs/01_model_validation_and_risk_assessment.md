# Model Validation and Risk Assessment

## Purpose

This document records the model-risk review performed for the Financial Transaction Risk Monitoring project.

The project uses a synthetic transaction dataset and is intended as a portfolio proof of concept. The purpose of this review is not to maximise a headline score. It is to determine whether the observed model performance represents genuine out-of-time predictive signal or whether it is driven by a synthetic target-generation rule.

---

## Dataset and Evaluation Design

The cleaned dataset contains:

- 50,000 transactions;
- 26 columns after time-feature creation;
- transaction timestamps from 1 January 2023 to 31 December 2023;
- 16,067 fraud-labelled transactions;
- an overall synthetic fraud rate of 32.13%.

The final validation design uses chronological splits:

| Split | Rows | Period | Fraud rate |
|---|---:|---|---:|
| Train | 30,000 | 2023-01-01 to 2023-08-07 | 32.14% |
| Validation | 10,000 | 2023-08-07 to 2023-10-19 | 31.97% |
| Test | 10,000 | 2023-10-19 to 2023-12-31 | 32.28% |

The split policy is:

- Train is used to fit candidate models.
- Validation is used to compare models and determine thresholds.
- Test is used once for final out-of-time evaluation.
- `risk_score` is excluded because its generation process is not transparent.
- `transaction_id`, `user_id`, and the raw timestamp are excluded from model inputs.
- `failed_transaction_count_7d` is audited separately because it behaves like a synthetic target-generation proxy.

---

## Finding 1: Deterministic Synthetic Proxy

The audit found the following fraud rates:

| Failed transactions in previous 7 days | Transactions | Fraud rate |
|---:|---:|---:|
| 0 | 10,014 | 15.40% |
| 1 | 9,919 | 15.44% |
| 2 | 9,897 | 14.54% |
| 3 | 10,216 | 15.67% |
| 4 | 9,954 | 100.00% |

Every transaction with `failed_transaction_count_7d == 4` is labelled as fraud.

A simple rule:

```text
failed_transaction_count_7d >= 4
```

produces the following out-of-time Test results:

| Metric | Result |
|---|---:|
| Review rate | 20.27% |
| Precision | 100.00% |
| Recall | 62.79% |
| F1 | 77.15% |
| False positives | 0 |
| False negatives | 1,201 |

This pattern is stable in Train, Validation, and Test. Therefore, the strong performance of the original Random Forest and XGBoost models is primarily explained by this deterministic synthetic rule rather than broad multivariate fraud-learning capability.

This field is not necessarily a software leakage bug. It is better described as a synthetic target-generation proxy.

---

## Finding 2: Feature Ablation

Two feature sets were compared:

1. **Benchmark feature set**
   - excludes `risk_score`;
   - includes `failed_transaction_count_7d`.

2. **Conservative feature set**
   - excludes `risk_score`;
   - excludes `failed_transaction_count_7d`.

On Validation, benchmark models achieved PR-AUC values around 0.80 because they learned the deterministic failed-count rule.

After removing the proxy, performance fell to approximately random levels:

| Conservative model | Validation ROC-AUC | Validation PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.4977 | 0.3204 |
| Random Forest | 0.4950 | 0.3177 |
| XGBoost | 0.5030 | 0.3230 |

The conservative XGBoost candidate was selected using Validation only and then evaluated once on Test:

| Metric | Test result |
|---|---:|
| Fraud-rate baseline | 32.28% |
| ROC-AUC | 0.4921 |
| PR-AUC | 0.3204 |
| Precision at approximately 20% review capacity | 31.71% |
| Recall | 19.95% |

The model did not outperform the prevalence baseline in a meaningful way.

---

## Finding 3: Residual Fraud Signal

The deterministic rule-covered transactions were removed, leaving only:

```text
failed_transaction_count_7d < 4
```

Residual fraud rates remained stable:

| Split | Rows | Fraud rate |
|---|---:|---:|
| Train | 24,026 | 15.27% |
| Validation | 8,047 | 15.46% |
| Test | 7,973 | 15.06% |

The strongest individual residual feature effects were very small. For example:

- card-type fraud-rate range: approximately 1.69 percentage points;
- transaction-type fraud-rate range: approximately 1.55 percentage points;
- authentication-method fraud-rate range: approximately 1.39 percentage points;
- best numeric single-feature AUC: approximately 0.51.

Residual candidate models produced:

| Model | Validation ROC-AUC | Validation PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.5111 | 0.1586 |
| Random Forest | 0.4987 | 0.1564 |
| XGBoost | 0.4960 | 0.1525 |

Logistic Regression was retained as the diagnostic candidate based on Validation PR-AUC. Its independent Test results were:

| Metric | Test result |
|---|---:|
| Residual fraud-rate baseline | 15.06% |
| ROC-AUC | 0.4875 |
| PR-AUC | 0.1458 |
| Relative PR-AUC uplift | -3.20% |

The residual model performed below the prevalence baseline and does not provide reliable out-of-time risk ranking.

---

## Two-Layer Strategy Review

A two-layer framework was tested:

1. deterministic rule layer;
2. residual model ranking layer.

Test results:

| Strategy | Review rate | Precision | Recall |
|---|---:|---:|---:|
| Rule only | 20.27% | 100.00% | 62.79% |
| Rule + residual top 5% | 22.91% | 90.05% | 63.91% |
| Rule + residual top 10% | 25.95% | 80.81% | 64.96% |
| Rule + residual top 20% | 32.31% | 68.21% | 68.28% |

The additional residual reviews had fraud hit rates between approximately 12.32% and 14.70%, which did not exceed the residual fraud baseline of 15.06%.

Therefore, the residual model increases review workload and reduces precision without producing reliable incremental targeting value.

---

## Deployment Decision

### Decision

```text
DO NOT DEPLOY THE RESIDUAL MACHINE-LEARNING MODEL
```

### Rationale

The residual model fails both minimum acceptance criteria:

- Test ROC-AUC must be at least 0.55.
- Relative Test PR-AUC uplift must be at least 10%.

Observed results:

- Test ROC-AUC: 0.4875.
- Relative Test PR-AUC uplift: -3.20%.

### Approved Portfolio Positioning

The project should be presented as:

> An end-to-end financial transaction risk-monitoring prototype with model-risk auditing, temporal validation, proxy-feature detection, threshold analysis, API serving, dashboard monitoring, and Docker-based deployment.

The deterministic rule may remain in the project as a transparent synthetic-data demonstration. It must not be described as evidence of production banking performance.

The residual model may remain as a diagnostic artifact. It must not drive automated transaction decisions.

---

## Governance Status

| Component | Status | Intended use |
|---|---|---|
| Synthetic failed-count rule | Demonstration only | Shows rule detection and workload trade-offs |
| Original Random Forest benchmark | Not production eligible | Historical benchmark and engineering demonstration |
| Conservative XGBoost | Rejected | No meaningful out-of-time signal |
| Residual Logistic Regression | Diagnostic only | Demonstrates model validation and rejection |
| FastAPI and Streamlit | Engineering prototype | Demonstrates service and monitoring architecture |
| Automatic fraud blocking | Not approved | Requires real data, calibration, controls, and governance |

---

## Required Data Improvements

A deployable fraud model would require more informative and realistically generated features, such as:

- verified historical user behaviour;
- device and account velocity features;
- merchant and counterparty history;
- geographically consistent transaction sequences;
- rolling user-level aggregates computed only from past observations;
- delayed fraud labels;
- realistic class imbalance;
- concept drift and fraud-pattern changes;
- investigation outcomes and financial loss severity.

The synthetic target should not be generated by a deterministic single-field condition.

---

## Limitations

- The fraud rate is much higher than in many real-world financial systems.
- The dataset contains no real customer or banking information.
- The target-generation logic creates an unrealistic deterministic proxy.
- User overlap across chronological splits is high, so the evaluation mainly reflects future transactions from previously observed users.
- The project does not establish cold-start performance for unseen users.
- Model probabilities are not calibrated for production use.
- No automated blocking, customer-impact decision, or compliance conclusion should be based on this prototype.

---

## Reproducibility

Run the validation scripts in this order:

```powershell
python .\src\00_feature_leakage_audit.py
python .\src\04b_temporal_ablation_models.py
python .\src\04c_residual_signal_analysis.py
```

Key outputs include:

```text
outputs/failed_count_rule_baseline_metrics.csv
outputs/temporal_ablation_validation_results.csv
outputs/temporal_ablation_final_test_metrics.csv
outputs/residual_validation_model_comparison.csv
outputs/residual_independent_test_metrics.csv
outputs/two_layer_test_strategy.csv
outputs/residual_signal_decision.json
```

---

## Final Conclusion

The most important result of this project is not a high model score.

The project demonstrates that a responsible data scientist must challenge suspicious performance, identify synthetic proxy behaviour, protect the final test set, compare against simple rules, evaluate out-of-time stability, and reject a model when it does not deliver genuine incremental value.
