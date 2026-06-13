from pathlib import Path
import sqlite3
import pandas as pd


# ============================================================
# 0. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "fraud_data_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
SQL_DIR = BASE_DIR / "sql"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SQL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = PROCESSED_DIR / "fraud_monitoring.db"


# ============================================================
# 1. 读取 cleaned data
# ============================================================

df = pd.read_csv(DATA_FILE)

# SQLite 对 datetime 没有专门类型，所以这里把 timestamp 转成字符串保存
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

print("\n========== CLEANED DATA LOADED ==========")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# 2. 写入 SQLite 数据库
# ============================================================

conn = sqlite3.connect(DB_FILE)

# 把 DataFrame 写入 SQLite，表名叫 transactions
df.to_sql("transactions", conn, if_exists="replace", index=False)

# 创建索引：加速后续 SQL 查询
conn.execute("CREATE INDEX IF NOT EXISTS idx_transaction_id ON transactions(transaction_id);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON transactions(user_id);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON transactions(timestamp);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_fraud_label ON transactions(fraud_label);")
conn.commit()

print("\n========== SQLITE DATABASE CREATED ==========")
print("Database saved to:", DB_FILE)
print("Table created: transactions")


# ============================================================
# 3. 工具函数：运行 SQL 并保存结果
# ============================================================

def run_sql_and_save(query, output_csv_name, sql_file_name):
    """
    运行一段 SQL，把结果保存成 CSV，同时把 SQL 语句也保存到 sql/ 文件夹。
    """

    result = pd.read_sql_query(query, conn)

    output_csv_path = OUTPUT_DIR / output_csv_name
    sql_file_path = SQL_DIR / sql_file_name

    result.to_csv(output_csv_path, index=False)
    sql_file_path.write_text(query, encoding="utf-8")

    print(f"\nSaved CSV: {output_csv_path}")
    print(f"Saved SQL: {sql_file_path}")
    print(result.head())

    return result


# ============================================================
# 4. SQL 查询 1：整体交易与 fraud 概览
# ============================================================

query_fraud_summary = """
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
"""

fraud_summary = run_sql_and_save(
    query=query_fraud_summary,
    output_csv_name="sql_database_fraud_summary.csv",
    sql_file_name="01_database_fraud_summary.sql",
)


# ============================================================
# 5. SQL 查询 2：用户级风险行为特征
# ============================================================

query_user_features = """
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
"""

user_features = run_sql_and_save(
    query=query_user_features,
    output_csv_name="sql_user_risk_features.csv",
    sql_file_name="02_user_risk_features.sql",
)


# ============================================================
# 6. SQL 查询 3：筛选规则型高风险交易
# ============================================================

# 这里先用 Python 计算高金额和远距离的分位数阈值
# 95% 分位数意思是：超过这个值的交易金额 / 交易距离属于相对极端的前 5%
high_amount_threshold = df["transaction_amount"].quantile(0.95)
high_distance_threshold = df["transaction_distance"].quantile(0.95)

print("\n========== RULE THRESHOLDS ==========")
print(f"High amount threshold 95%: {high_amount_threshold:.2f}")
print(f"High distance threshold 95%: {high_distance_threshold:.2f}")


query_high_risk_transactions = f"""
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
        + CASE WHEN transaction_amount >= {high_amount_threshold:.6f} THEN 1 ELSE 0 END
        + CASE WHEN transaction_distance >= {high_distance_threshold:.6f} THEN 1 ELSE 0 END
    ) AS rule_risk_score

FROM transactions

WHERE
    ip_address_flag = 1
    OR previous_fraudulent_activity = 1
    OR failed_transaction_count_7d >= 3
    OR daily_transaction_count >= 12
    OR transaction_amount >= {high_amount_threshold:.6f}
    OR transaction_distance >= {high_distance_threshold:.6f}

ORDER BY rule_risk_score DESC, transaction_amount DESC;
"""

high_risk_transactions = run_sql_and_save(
    query=query_high_risk_transactions,
    output_csv_name="sql_high_risk_transactions_rule_based.csv",
    sql_file_name="03_high_risk_transactions_rule_based.sql",
)


# ============================================================
# 7. SQL 查询 4：按交易类型看 fraud rate
# ============================================================

query_transaction_type_summary = """
SELECT
    transaction_type,
    COUNT(*) AS total_transactions,
    SUM(fraud_label) AS fraud_transactions,
    ROUND(AVG(fraud_label) * 100, 2) AS fraud_rate_pct,
    ROUND(AVG(transaction_amount), 2) AS avg_transaction_amount
FROM transactions
GROUP BY transaction_type
ORDER BY fraud_rate_pct DESC;
"""

transaction_type_summary = run_sql_and_save(
    query=query_transaction_type_summary,
    output_csv_name="sql_transaction_type_fraud_summary.csv",
    sql_file_name="04_transaction_type_fraud_summary.sql",
)


# ============================================================
# 8. SQL 查询 5：按小时看 fraud rate
# ============================================================

query_hourly_summary = """
SELECT
    hour,
    COUNT(*) AS total_transactions,
    SUM(fraud_label) AS fraud_transactions,
    ROUND(AVG(fraud_label) * 100, 2) AS fraud_rate_pct,
    ROUND(AVG(transaction_amount), 2) AS avg_transaction_amount
FROM transactions
GROUP BY hour
ORDER BY hour;
"""

hourly_summary = run_sql_and_save(
    query=query_hourly_summary,
    output_csv_name="sql_hourly_fraud_summary.csv",
    sql_file_name="05_hourly_fraud_summary.sql",
)


# ============================================================
# 9. 规则型高风险交易效果检查
# ============================================================

high_risk_count = len(high_risk_transactions)
high_risk_fraud_count = int(high_risk_transactions["fraud_label"].sum())
high_risk_fraud_rate = high_risk_transactions["fraud_label"].mean() * 100

overall_fraud_rate = df["fraud_label"].mean() * 100

print("\n========== RULE-BASED HIGH RISK CHECK ==========")
print("High-risk transactions selected:", high_risk_count)
print("Fraud transactions within high-risk group:", high_risk_fraud_count)
print(f"High-risk group fraud rate: {high_risk_fraud_rate:.2f}%")
print(f"Overall fraud rate: {overall_fraud_rate:.2f}%")


# ============================================================
# 10. 保存 SQL 特征工程总结指标
# ============================================================

summary_metrics = pd.DataFrame(
    [
        ["database_path", str(DB_FILE)],
        ["transaction_table", "transactions"],
        ["total_transactions", len(df)],
        ["unique_users", df["user_id"].nunique()],
        ["overall_fraud_rate_pct", round(overall_fraud_rate, 2)],
        ["user_feature_rows", len(user_features)],
        ["high_risk_transactions_selected", high_risk_count],
        ["high_risk_group_fraud_rate_pct", round(high_risk_fraud_rate, 2)],
        ["high_amount_threshold_95pct", round(high_amount_threshold, 2)],
        ["high_distance_threshold_95pct", round(high_distance_threshold, 2)],
    ],
    columns=["metric", "value"],
)

summary_metrics.to_csv(OUTPUT_DIR / "sql_feature_engineering_summary_metrics.csv", index=False)

print("\n========== SQL FEATURE ENGINEERING SUMMARY ==========")
print(summary_metrics)


# ============================================================
# 11. 关闭数据库连接
# ============================================================

conn.close()

print("\nSQL feature engineering completed successfully.")