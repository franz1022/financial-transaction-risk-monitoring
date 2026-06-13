from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib

# 终端运行时不弹出窗口，直接保存图片
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 0. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "fraud_data_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 读取清洗后的数据
# ============================================================

df = pd.read_csv(DATA_FILE)

# 时间列重新转成 datetime，方便后面按小时、月份分析
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

print("\n========== CLEANED DATA LOADED ==========")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# 2. 基础 fraud 分布
# ============================================================

total_rows = len(df)
fraud_count = int(df["fraud_label"].sum())
normal_count = total_rows - fraud_count
fraud_rate = df["fraud_label"].mean() * 100

print("\n========== OVERALL FRAUD SUMMARY ==========")
print("Total transactions:", total_rows)
print("Normal transactions:", normal_count)
print("Fraud transactions:", fraud_count)
print(f"Fraud rate: {fraud_rate:.2f}%")


class_distribution = (
    df["fraud_label"]
    .value_counts()
    .sort_index()
    .rename_axis("fraud_label")
    .reset_index(name="count")
)

class_distribution["percentage"] = class_distribution["count"] / total_rows * 100
class_distribution.to_csv(OUTPUT_DIR / "fraud_class_distribution.csv", index=False)

# 画图：正常交易 vs 欺诈交易数量
plt.figure(figsize=(7, 5))
plt.bar(["Normal (0)", "Fraud (1)"], class_distribution["count"])
plt.title("Fraud Class Distribution")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

for i, value in enumerate(class_distribution["count"]):
    plt.text(i, value, str(value), ha="center", va="bottom")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "01_fraud_class_distribution.png", dpi=160)
plt.close()


# ============================================================
# 3. 通用函数：按类别字段计算 fraud rate
# ============================================================

def fraud_rate_by_category(df, category_col, output_name, figure_name, title, top_n=None):
    """
    按某个类别字段计算欺诈率。

    fraud_rate = fraud交易数 / 总交易数

    例如：
    transaction_type = Online
    里面一共有 1000 笔交易，其中 300 笔 fraud
    那么 fraud_rate = 300 / 1000 = 30%
    """

    summary = (
        df.groupby(category_col)
        .agg(
            total_transactions=("fraud_label", "size"),
            fraud_transactions=("fraud_label", "sum"),
            fraud_rate=("fraud_label", "mean"),
        )
        .reset_index()
    )

    summary["fraud_rate_pct"] = summary["fraud_rate"] * 100
    summary = summary.sort_values("fraud_rate_pct", ascending=False)

    summary.to_csv(OUTPUT_DIR / output_name, index=False)

    plot_data = summary.copy()
    if top_n is not None:
        plot_data = plot_data.head(top_n)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_data[category_col].astype(str), plot_data["fraud_rate_pct"])
    plt.title(title)
    plt.xlabel(category_col)
    plt.ylabel("Fraud Rate (%)")
    plt.xticks(rotation=35, ha="right")

    for i, value in enumerate(plot_data["fraud_rate_pct"]):
        plt.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / figure_name, dpi=160)
    plt.close()

    return summary


# ============================================================
# 4. 按交易类型、设备、地点、商户类别分析 fraud rate
# ============================================================

transaction_type_summary = fraud_rate_by_category(
    df=df,
    category_col="transaction_type",
    output_name="fraud_rate_by_transaction_type.csv",
    figure_name="02_fraud_rate_by_transaction_type.png",
    title="Fraud Rate by Transaction Type",
)

device_type_summary = fraud_rate_by_category(
    df=df,
    category_col="device_type",
    output_name="fraud_rate_by_device_type.csv",
    figure_name="03_fraud_rate_by_device_type.png",
    title="Fraud Rate by Device Type",
)

location_summary = fraud_rate_by_category(
    df=df,
    category_col="location",
    output_name="fraud_rate_by_location.csv",
    figure_name="04_fraud_rate_by_location.png",
    title="Fraud Rate by Location",
)

merchant_summary = fraud_rate_by_category(
    df=df,
    category_col="merchant_category",
    output_name="fraud_rate_by_merchant_category.csv",
    figure_name="05_fraud_rate_by_merchant_category.png",
    title="Fraud Rate by Merchant Category",
)

auth_summary = fraud_rate_by_category(
    df=df,
    category_col="authentication_method",
    output_name="fraud_rate_by_authentication_method.csv",
    figure_name="06_fraud_rate_by_authentication_method.png",
    title="Fraud Rate by Authentication Method",
)


# ============================================================
# 5. 按小时分析 fraud rate
# ============================================================

hourly_summary = (
    df.groupby("hour")
    .agg(
        total_transactions=("fraud_label", "size"),
        fraud_transactions=("fraud_label", "sum"),
        fraud_rate=("fraud_label", "mean"),
    )
    .reset_index()
)

hourly_summary["fraud_rate_pct"] = hourly_summary["fraud_rate"] * 100
hourly_summary.to_csv(OUTPUT_DIR / "fraud_rate_by_hour.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(hourly_summary["hour"], hourly_summary["fraud_rate_pct"], marker="o")
plt.title("Fraud Rate by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Fraud Rate (%)")
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "07_fraud_rate_by_hour.png", dpi=160)
plt.close()


# ============================================================
# 6. 按月份分析 fraud rate
# ============================================================

monthly_summary = (
    df.groupby("month")
    .agg(
        total_transactions=("fraud_label", "size"),
        fraud_transactions=("fraud_label", "sum"),
        fraud_rate=("fraud_label", "mean"),
    )
    .reset_index()
)

monthly_summary["fraud_rate_pct"] = monthly_summary["fraud_rate"] * 100
monthly_summary.to_csv(OUTPUT_DIR / "fraud_rate_by_month.csv", index=False)

plt.figure(figsize=(9, 5))
plt.plot(monthly_summary["month"], monthly_summary["fraud_rate_pct"], marker="o")
plt.title("Fraud Rate by Month")
plt.xlabel("Month")
plt.ylabel("Fraud Rate (%)")
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "08_fraud_rate_by_month.png", dpi=160)
plt.close()


# ============================================================
# 7. 交易金额分布：Normal vs Fraud
# ============================================================

normal_amount = df.loc[df["fraud_label"] == 0, "transaction_amount"]
fraud_amount = df.loc[df["fraud_label"] == 1, "transaction_amount"]

plt.figure(figsize=(9, 5))
plt.hist(normal_amount, bins=50, alpha=0.6, label="Normal", density=True)
plt.hist(fraud_amount, bins=50, alpha=0.6, label="Fraud", density=True)
plt.title("Transaction Amount Distribution by Fraud Label")
plt.xlabel("Transaction Amount")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "09_amount_distribution_by_fraud_label.png", dpi=160)
plt.close()


# 金额描述统计
amount_summary = (
    df.groupby("fraud_label")["transaction_amount"]
    .describe()
    .reset_index()
)

amount_summary.to_csv(OUTPUT_DIR / "amount_summary_by_fraud_label.csv", index=False)


# ============================================================
# 8. 关键二元风险字段分析
# ============================================================

binary_risk_cols = [
    "ip_address_flag",
    "previous_fraudulent_activity",
    "is_weekend",
]

binary_risk_results = []

for col in binary_risk_cols:
    temp = (
        df.groupby(col)
        .agg(
            total_transactions=("fraud_label", "size"),
            fraud_transactions=("fraud_label", "sum"),
            fraud_rate=("fraud_label", "mean"),
        )
        .reset_index()
    )

    temp["fraud_rate_pct"] = temp["fraud_rate"] * 100
    temp["risk_factor"] = col
    temp = temp.rename(columns={col: "factor_value"})

    binary_risk_results.append(temp)

binary_risk_summary = pd.concat(binary_risk_results, ignore_index=True)
binary_risk_summary.to_csv(OUTPUT_DIR / "binary_risk_factor_summary.csv", index=False)

# 画图：二元风险因素为 1 时的 fraud rate
risk_factor_plot = binary_risk_summary[binary_risk_summary["factor_value"] == 1].copy()

plt.figure(figsize=(9, 5))
plt.bar(risk_factor_plot["risk_factor"], risk_factor_plot["fraud_rate_pct"])
plt.title("Fraud Rate When Risk Factor = 1")
plt.xlabel("Risk Factor")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=25, ha="right")

for i, value in enumerate(risk_factor_plot["fraud_rate_pct"]):
    plt.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "10_binary_risk_factors_fraud_rate.png", dpi=160)
plt.close()


# ============================================================
# 9. 近 7 天失败交易次数与 fraud rate
# ============================================================

failed_txn_summary = (
    df.groupby("failed_transaction_count_7d")
    .agg(
        total_transactions=("fraud_label", "size"),
        fraud_transactions=("fraud_label", "sum"),
        fraud_rate=("fraud_label", "mean"),
    )
    .reset_index()
)

failed_txn_summary["fraud_rate_pct"] = failed_txn_summary["fraud_rate"] * 100
failed_txn_summary.to_csv(OUTPUT_DIR / "fraud_rate_by_failed_transaction_count_7d.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(
    failed_txn_summary["failed_transaction_count_7d"].astype(str),
    failed_txn_summary["fraud_rate_pct"],
)
plt.title("Fraud Rate by Failed Transaction Count in Last 7 Days")
plt.xlabel("Failed Transaction Count in Last 7 Days")
plt.ylabel("Fraud Rate (%)")

for i, value in enumerate(failed_txn_summary["fraud_rate_pct"]):
    plt.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "11_fraud_rate_by_failed_transaction_count_7d.png", dpi=160)
plt.close()


# ============================================================
# 10. Daily transaction count 与 fraud rate
# ============================================================

daily_count_summary = (
    df.groupby("daily_transaction_count")
    .agg(
        total_transactions=("fraud_label", "size"),
        fraud_transactions=("fraud_label", "sum"),
        fraud_rate=("fraud_label", "mean"),
    )
    .reset_index()
)

daily_count_summary["fraud_rate_pct"] = daily_count_summary["fraud_rate"] * 100
daily_count_summary.to_csv(OUTPUT_DIR / "fraud_rate_by_daily_transaction_count.csv", index=False)

plt.figure(figsize=(9, 5))
plt.plot(
    daily_count_summary["daily_transaction_count"],
    daily_count_summary["fraud_rate_pct"],
    marker="o",
)
plt.title("Fraud Rate by Daily Transaction Count")
plt.xlabel("Daily Transaction Count")
plt.ylabel("Fraud Rate (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "12_fraud_rate_by_daily_transaction_count.png", dpi=160)
plt.close()


# ============================================================
# 11. 注意：已有 risk_score 只用于 EDA，不用于后续模型训练
# ============================================================

risk_score_summary = (
    df.groupby("fraud_label")["risk_score"]
    .describe()
    .reset_index()
)

risk_score_summary.to_csv(OUTPUT_DIR / "existing_risk_score_summary_by_fraud_label.csv", index=False)

plt.figure(figsize=(9, 5))
plt.hist(
    df.loc[df["fraud_label"] == 0, "risk_score"],
    bins=40,
    alpha=0.6,
    label="Normal",
    density=True,
)
plt.hist(
    df.loc[df["fraud_label"] == 1, "risk_score"],
    bins=40,
    alpha=0.6,
    label="Fraud",
    density=True,
)
plt.title("Existing Risk Score Distribution by Fraud Label")
plt.xlabel("Existing Risk Score")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "13_existing_risk_score_distribution.png", dpi=160)
plt.close()


# ============================================================
# 12. EDA 总结指标
# ============================================================

def get_top_category(summary_df, category_col):
    row = summary_df.sort_values("fraud_rate_pct", ascending=False).iloc[0]
    return row[category_col], row["fraud_rate_pct"]


top_transaction_type, top_transaction_type_rate = get_top_category(
    transaction_type_summary, "transaction_type"
)
top_device_type, top_device_type_rate = get_top_category(
    device_type_summary, "device_type"
)
top_location, top_location_rate = get_top_category(
    location_summary, "location"
)
top_merchant, top_merchant_rate = get_top_category(
    merchant_summary, "merchant_category"
)
top_auth, top_auth_rate = get_top_category(
    auth_summary, "authentication_method"
)

highest_hour_row = hourly_summary.sort_values("fraud_rate_pct", ascending=False).iloc[0]
highest_month_row = monthly_summary.sort_values("fraud_rate_pct", ascending=False).iloc[0]

eda_summary_metrics = pd.DataFrame(
    [
        ["total_transactions", total_rows],
        ["normal_transactions", normal_count],
        ["fraud_transactions", fraud_count],
        ["overall_fraud_rate_pct", round(fraud_rate, 2)],
        ["top_transaction_type_by_fraud_rate", top_transaction_type],
        ["top_transaction_type_fraud_rate_pct", round(top_transaction_type_rate, 2)],
        ["top_device_type_by_fraud_rate", top_device_type],
        ["top_device_type_fraud_rate_pct", round(top_device_type_rate, 2)],
        ["top_location_by_fraud_rate", top_location],
        ["top_location_fraud_rate_pct", round(top_location_rate, 2)],
        ["top_merchant_category_by_fraud_rate", top_merchant],
        ["top_merchant_category_fraud_rate_pct", round(top_merchant_rate, 2)],
        ["top_authentication_method_by_fraud_rate", top_auth],
        ["top_authentication_method_fraud_rate_pct", round(top_auth_rate, 2)],
        ["highest_fraud_rate_hour", int(highest_hour_row["hour"])],
        ["highest_hour_fraud_rate_pct", round(highest_hour_row["fraud_rate_pct"], 2)],
        ["highest_fraud_rate_month", int(highest_month_row["month"])],
        ["highest_month_fraud_rate_pct", round(highest_month_row["fraud_rate_pct"], 2)],
    ],
    columns=["metric", "value"],
)

eda_summary_metrics.to_csv(OUTPUT_DIR / "eda_summary_metrics.csv", index=False)


print("\n========== EDA SUMMARY ==========")
print(eda_summary_metrics)

print("\n========== FILES SAVED ==========")
print("Summary tables saved to:", OUTPUT_DIR)
print("Figures saved to:", FIGURE_DIR)

print("\nEDA fraud analysis completed successfully.")