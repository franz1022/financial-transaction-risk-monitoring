# Risk Scoring Business Report

## Objective

The objective of this step is to convert model-based fraud probabilities into business-friendly risk levels and recommended actions.

Instead of only returning a binary fraud prediction, the system assigns each transaction to one of three risk levels:

| Risk Level | Fraud Probability Range | Recommended Action |
|---|---:|---|
| Low | < 0.20 | Auto Approve |
| Medium | 0.20 - 0.70 | Monitor / Secondary Check |
| High | >= 0.70 | Manual Review / Alert |

---

## Dataset Summary

| Metric | Value |
|---|---:|
| Scored transactions | 10000 |
| Fraud transactions | 3213 |
| Overall fraud rate | 32.13% |

---

## Risk Level Distribution

| Risk Level | Transaction Count | Transaction Share | Fraud Rate |
|---|---:|---:|---:|
| Low | 6405 | 64.05% | 15.04% |
| Medium | 1610 | 16.10% | 16.46% |
| High | 1985 | 19.85% | 100.00% |


---

## High-risk Transaction Monitoring

At the high-risk threshold of 0.70:

| Metric | Value |
|---|---:|
| High-risk transactions | 1985 |
| High-risk share | 19.85% |
| Fraud transactions captured in high-risk group | 1985 |
| High-risk precision | 100.00% |
| High-risk recall | 61.78% |

---

## Business Interpretation

The high-risk group is designed for manual review or immediate fraud alert.  
A high threshold produces highly reliable fraud alerts and reduces false positives, but it may miss some fraudulent transactions.

The medium-risk group can be used for secondary checks, monitoring, or additional authentication.  
The low-risk group can be automatically approved to reduce operational workload.

---

## Recommended Usage

- Use **High Risk** for manual review or real-time alerting.
- Use **Medium Risk** for additional verification, such as OTP or device confirmation.
- Use **Low Risk** for automatic approval.
- Adjust thresholds according to risk appetite and manual review capacity.
