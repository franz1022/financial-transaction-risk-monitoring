
SELECT
    transaction_id,
    user_id,
    timestamp,
    transaction_amount,
    transaction_type,
    account_balance,
    device_type,
    location,
    merchant_category,
    authentication_method,

    ip_address_flag,
    previous_fraudulent_activity,
    daily_transaction_count,
    failed_transaction_count_7d,
    transaction_distance,

    fraud_label,

    (
        ip_address_flag
        + previous_fraudulent_activity
        + CASE WHEN failed_transaction_count_7d >= 3 THEN 1 ELSE 0 END
        + CASE WHEN daily_transaction_count >= 12 THEN 1 ELSE 0 END
        + CASE WHEN transaction_amount >= 294.303500 THEN 1 ELSE 0 END
        + CASE WHEN transaction_distance >= 4756.925500 THEN 1 ELSE 0 END
    ) AS rule_risk_score

FROM transactions

WHERE
    ip_address_flag = 1
    OR previous_fraudulent_activity = 1
    OR failed_transaction_count_7d >= 3
    OR daily_transaction_count >= 12
    OR transaction_amount >= 294.303500
    OR transaction_distance >= 4756.925500

ORDER BY rule_risk_score DESC, transaction_amount DESC;
