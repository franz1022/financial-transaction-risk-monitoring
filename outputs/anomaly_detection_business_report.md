# Anomaly Detection Business Report

## Objective

This step adds an unsupervised anomaly detection layer to the transaction risk monitoring system.

Unlike supervised fraud classification models, Isolation Forest does not use fraud labels during training. It learns the general pattern of normal transaction behavior and assigns higher anomaly scores to unusual transactions.

This is useful when:

- fraud labels are incomplete
- fraud labels are delayed
- new fraud patterns have not yet been labeled
- the business wants an early warning system for unusual transactions

---

## Model

| Item | Value |
|---|---|
| Model | Isolation Forest |
| Training mode | Unsupervised |
| Features used | Transaction, device, location, merchant, behavior, and time features |
| Excluded columns | transaction_id, user_id, timestamp, risk_score, fraud_label |

The existing risk_score was excluded to avoid target leakage.

---

## Continuous Anomaly Score Performance

| Metric | Value |
|---|---:|
| ROC-AUC | 53.93% |
| PR-AUC | 35.14% |

---

## Best Review-rate Strategy by F1-score

| Metric | Value |
|---|---:|
| Review rate | 30.00% |
| Transactions reviewed | 3000 |
| Fraud captured | 1083 |
| False alerts | 1917 |
| Precision | 36.10% |
| Recall | 33.71% |
| F1-score | 34.86% |

---

## Business Interpretation

Isolation Forest is not expected to replace supervised fraud models when reliable fraud labels are available.  
Instead, it works as a complementary early-warning layer for suspicious and unusual transaction patterns.

The business can choose a review rate, such as top 5%, 10%, or 20% most anomalous transactions, depending on manual review capacity.

---

## Recommended Usage

- Use supervised models for known fraud pattern detection.
- Use Isolation Forest to detect unusual transactions and potential new fraud patterns.
- Combine anomaly scores with model-based risk scores for a multi-layer risk monitoring system.
