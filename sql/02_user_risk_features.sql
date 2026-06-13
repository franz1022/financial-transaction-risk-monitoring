
SELECT
    user_id,

    COUNT(*) AS total_transactions,

    ROUND(SUM(transaction_amount), 2) AS total_transaction_amount,
    ROUND(AVG(transaction_amount), 2) AS avg_transaction_amount,
    ROUND(MAX(transaction_amount), 2) AS max_transaction_amount,

    ROUND(AVG(account_balance), 2) AS avg_account_balance,

    SUM(fraud_label) AS fraud_transactions,
    ROUND(AVG(fraud_label) * 100, 2) AS fraud_rate_pct,

    ROUND(AVG(ip_address_flag) * 100, 2) AS ip_flag_rate_pct,
    ROUND(AVG(previous_fraudulent_activity) * 100, 2) AS previous_fraud_activity_rate_pct,

    ROUND(AVG(daily_transaction_count), 2) AS avg_daily_transaction_count,
    ROUND(AVG(failed_transaction_count_7d), 2) AS avg_failed_transaction_count_7d,
    ROUND(AVG(transaction_distance), 2) AS avg_transaction_distance,

    COUNT(DISTINCT location) AS distinct_locations,
    COUNT(DISTINCT device_type) AS distinct_device_types,
    COUNT(DISTINCT merchant_category) AS distinct_merchant_categories,

    MIN(timestamp) AS first_transaction_time,
    MAX(timestamp) AS last_transaction_time

FROM transactions
GROUP BY user_id
ORDER BY fraud_rate_pct DESC, total_transactions DESC;
