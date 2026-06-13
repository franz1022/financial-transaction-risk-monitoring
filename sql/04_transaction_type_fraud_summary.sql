
SELECT
    transaction_type,
    COUNT(*) AS total_transactions,
    SUM(fraud_label) AS fraud_transactions,
    ROUND(AVG(fraud_label) * 100, 2) AS fraud_rate_pct,
    ROUND(AVG(transaction_amount), 2) AS avg_transaction_amount
FROM transactions
GROUP BY transaction_type
ORDER BY fraud_rate_pct DESC;
