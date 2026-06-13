
SELECT
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN fraud_label = 0 THEN 1 ELSE 0 END) AS normal_transactions,
    SUM(CASE WHEN fraud_label = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(AVG(fraud_label) * 100, 2) AS fraud_rate_pct,
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(DISTINCT transaction_type) AS transaction_type_count,
    COUNT(DISTINCT device_type) AS device_type_count,
    COUNT(DISTINCT location) AS location_count,
    COUNT(DISTINCT merchant_category) AS merchant_category_count
FROM transactions;
