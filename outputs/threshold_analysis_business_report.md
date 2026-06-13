# Threshold Analysis Business Report

## Objective

The goal of this analysis is to evaluate how different fraud probability thresholds affect fraud detection performance and manual review workload.

In fraud monitoring, the model does not only produce a 0/1 prediction. It also produces a fraud probability. By changing the decision threshold, the business team can balance two competing goals:

- catching more fraudulent transactions
- controlling false alarms and manual review workload

---

## Key Dataset Information

| Metric | Value |
|---|---:|
| Test transactions | 10000 |
| Fraud transactions | 3213 |
| Normal transactions | 6787 |
| Overall fraud rate | 32.13% |

---

## Recommended Threshold Based on F1-score

| Metric | Value |
|---|---:|
| Recommended threshold | 0.45 |
| Precision | 100.00% |
| Recall | 61.78% |
| F1-score | 76.38% |
| Manual review rate | 19.85% |
| Transactions flagged for review | 1985 |
| False positives | 0 |
| False negatives | 1228 |

---

## Business Interpretation

A lower threshold will flag more transactions as suspicious. This usually improves recall, meaning that more fraudulent transactions can be captured, but it also increases the number of false positives and manual review workload.

A higher threshold will only flag transactions with very high predicted fraud probability. This usually improves precision and reduces false positives, but it may miss more fraudulent transactions.

Therefore, the final threshold should depend on business capacity and risk appetite.

---

## Suggested Usage

- If the company wants to reduce fraud loss as much as possible, choose a lower threshold to increase recall.
- If the company has limited manual review resources, choose a higher threshold to reduce review workload.
- If the company wants a balanced strategy, use the threshold with the highest F1-score.
