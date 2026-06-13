from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib

# 终端运行，不弹出图片窗口，直接保存图片
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 0. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

PREDICTION_FILE = OUTPUT_DIR / "class_imbalance_model_predictions.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 读取模型预测结果
# ============================================================

df = pd.read_csv(PREDICTION_FILE)

print("\n========== PREDICTION DATA LOADED ==========")
print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())


# ============================================================
# 2. 选择模型概率列
# ============================================================

# 当前最佳模型是 Random Forest
TARGET_COL = "actual_fraud_label"
PROBA_COL = "random_forest_fraud_probability"

if TARGET_COL not in df.columns:
    raise ValueError(f"Cannot find target column: {TARGET_COL}")

if PROBA_COL not in df.columns:
    raise ValueError(f"Cannot find probability column: {PROBA_COL}")


# ============================================================
# 3. 生成模型风险分数
# ============================================================

# 模型输出的是 0-1 之间的 fraud probability
# 为了更像业务系统，我们把它乘以 100，变成 0-100 的 risk score
df["model_risk_score"] = df[PROBA_COL] * 100

# 创建一个测试集交易编号，方便后面展示
df["scored_transaction_id"] = [
    f"SCORED_TXN_{i:05d}" for i in range(1, len(df) + 1)
]


# ============================================================
# 4. 风险等级规则
# ============================================================

# 根据前一步 threshold analysis：
# < 0.20：低风险，自动通过
# 0.20 - 0.70：中风险，观察 / 二次校验
# >= 0.70：高风险，人工审核 / 风险预警

def assign_risk_level(probability):
    if probability < 0.20:
        return "Low"
    elif probability < 0.70:
        return "Medium"
    else:
        return "High"


def assign_action(risk_level):
    if risk_level == "Low":
        return "Auto Approve"
    elif risk_level == "Medium":
        return "Monitor / Secondary Check"
    else:
        return "Manual Review / Alert"


df["risk_level"] = df[PROBA_COL].apply(assign_risk_level)
df["recommended_action"] = df["risk_level"].apply(assign_action)


# ============================================================
# 5. 风险等级汇总
# ============================================================

total_transactions = len(df)
total_fraud = int(df[TARGET_COL].sum())
overall_fraud_rate = df[TARGET_COL].mean() * 100

risk_level_summary = (
    df.groupby("risk_level")
    .agg(
        transaction_count=("risk_level", "size"),
        fraud_count=(TARGET_COL, "sum"),
        avg_model_risk_score=("model_risk_score", "mean"),
        min_model_risk_score=("model_risk_score", "min"),
        max_model_risk_score=("model_risk_score", "max"),
    )
    .reset_index()
)

risk_level_summary["transaction_pct"] = (
    risk_level_summary["transaction_count"] / total_transactions * 100
)

risk_level_summary["fraud_rate_pct"] = (
    risk_level_summary["fraud_count"] / risk_level_summary["transaction_count"] * 100
)

risk_level_order = ["Low", "Medium", "High"]
risk_level_summary["risk_level"] = pd.Categorical(
    risk_level_summary["risk_level"],
    categories=risk_level_order,
    ordered=True,
)

risk_level_summary = risk_level_summary.sort_values("risk_level")

risk_level_summary.to_csv(
    OUTPUT_DIR / "risk_level_summary.csv",
    index=False,
)

print("\n========== RISK LEVEL SUMMARY ==========")
print(risk_level_summary)


# ============================================================
# 6. 推荐动作汇总
# ============================================================

action_summary = (
    df.groupby("recommended_action")
    .agg(
        transaction_count=("recommended_action", "size"),
        fraud_count=(TARGET_COL, "sum"),
        avg_model_risk_score=("model_risk_score", "mean"),
    )
    .reset_index()
)

action_summary["transaction_pct"] = (
    action_summary["transaction_count"] / total_transactions * 100
)

action_summary["fraud_rate_pct"] = (
    action_summary["fraud_count"] / action_summary["transaction_count"] * 100
)

action_summary.to_csv(
    OUTPUT_DIR / "risk_action_summary.csv",
    index=False,
)

print("\n========== ACTION SUMMARY ==========")
print(action_summary)


# ============================================================
# 7. 高风险交易清单
# ============================================================

high_risk_transactions = df[df["risk_level"] == "High"].copy()

high_risk_transactions = high_risk_transactions.sort_values(
    "model_risk_score",
    ascending=False,
)

high_risk_transactions.to_csv(
    OUTPUT_DIR / "high_risk_transactions_model_based.csv",
    index=False,
)

# 另外保存 Top 100 高风险交易，方便 README / dashboard 展示
top_100_high_risk = high_risk_transactions.head(100)

top_100_high_risk.to_csv(
    OUTPUT_DIR / "top_100_high_risk_transactions.csv",
    index=False,
)

print("\n========== HIGH RISK TRANSACTIONS ==========")
print("High risk transaction count:", len(high_risk_transactions))
print("Top 5 high risk transactions:")
print(
    top_100_high_risk[
        [
            "scored_transaction_id",
            PROBA_COL,
            "model_risk_score",
            "risk_level",
            "recommended_action",
            TARGET_COL,
        ]
    ].head()
)


# ============================================================
# 8. 低 / 中 / 高风险层级业务指标
# ============================================================

low_count = int((df["risk_level"] == "Low").sum())
medium_count = int((df["risk_level"] == "Medium").sum())
high_count = int((df["risk_level"] == "High").sum())

low_pct = low_count / total_transactions * 100
medium_pct = medium_count / total_transactions * 100
high_pct = high_count / total_transactions * 100

high_fraud_count = int(high_risk_transactions[TARGET_COL].sum())
high_precision = (
    high_fraud_count / high_count * 100
    if high_count > 0
    else 0
)

high_recall = (
    high_fraud_count / total_fraud * 100
    if total_fraud > 0
    else 0
)

summary_metrics = pd.DataFrame(
    [
        ["total_scored_transactions", total_transactions],
        ["total_fraud_transactions", total_fraud],
        ["overall_fraud_rate_pct", round(overall_fraud_rate, 2)],

        ["low_risk_transactions", low_count],
        ["low_risk_pct", round(low_pct, 2)],

        ["medium_risk_transactions", medium_count],
        ["medium_risk_pct", round(medium_pct, 2)],

        ["high_risk_transactions", high_count],
        ["high_risk_pct", round(high_pct, 2)],

        ["high_risk_fraud_count", high_fraud_count],
        ["high_risk_precision_pct", round(high_precision, 2)],
        ["high_risk_recall_pct", round(high_recall, 2)],

        ["low_threshold", 0.20],
        ["high_threshold", 0.70],
    ],
    columns=["metric", "value"],
)

summary_metrics.to_csv(
    OUTPUT_DIR / "risk_scoring_summary_metrics.csv",
    index=False,
)

print("\n========== RISK SCORING SUMMARY METRICS ==========")
print(summary_metrics)


# ============================================================
# 9. 图 1：风险等级交易数量
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    risk_level_summary["risk_level"].astype(str),
    risk_level_summary["transaction_count"],
)

plt.title("Transaction Count by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Number of Transactions")

for i, value in enumerate(risk_level_summary["transaction_count"]):
    plt.text(i, value, str(int(value)), ha="center", va="bottom")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "27_transaction_count_by_risk_level.png", dpi=160)
plt.close()


# ============================================================
# 10. 图 2：不同风险等级 fraud rate
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    risk_level_summary["risk_level"].astype(str),
    risk_level_summary["fraud_rate_pct"],
)

plt.title("Fraud Rate by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Fraud Rate (%)")

for i, value in enumerate(risk_level_summary["fraud_rate_pct"]):
    plt.text(i, value, f"{value:.1f}%", ha="center", va="bottom")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "28_fraud_rate_by_risk_level.png", dpi=160)
plt.close()


# ============================================================
# 11. 图 3：风险分数分布
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(
    df.loc[df[TARGET_COL] == 0, "model_risk_score"],
    bins=40,
    alpha=0.6,
    label="Normal",
    density=True,
)

plt.hist(
    df.loc[df[TARGET_COL] == 1, "model_risk_score"],
    bins=40,
    alpha=0.6,
    label="Fraud",
    density=True,
)

plt.axvline(20, linestyle="--", label="Low / Medium Threshold = 20")
plt.axvline(70, linestyle="--", label="Medium / High Threshold = 70")

plt.title("Model-based Risk Score Distribution")
plt.xlabel("Model-based Risk Score")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "29_model_risk_score_distribution.png", dpi=160)
plt.close()


# ============================================================
# 12. 图 4：推荐动作分布
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    action_summary["recommended_action"],
    action_summary["transaction_count"],
)

plt.title("Recommended Action Distribution")
plt.xlabel("Recommended Action")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=20, ha="right")

for i, value in enumerate(action_summary["transaction_count"]):
    plt.text(i, value, str(int(value)), ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "30_recommended_action_distribution.png", dpi=160)
plt.close()


# ============================================================
# 13. 生成 Risk Scoring 业务报告
# ============================================================

report_path = OUTPUT_DIR / "risk_scoring_business_report.md"

report = f"""# Risk Scoring Business Report

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
| Scored transactions | {total_transactions} |
| Fraud transactions | {total_fraud} |
| Overall fraud rate | {overall_fraud_rate:.2f}% |

---

## Risk Level Distribution

| Risk Level | Transaction Count | Transaction Share | Fraud Rate |
|---|---:|---:|---:|
"""

for _, row in risk_level_summary.iterrows():
    report += (
        f"| {row['risk_level']} | "
        f"{int(row['transaction_count'])} | "
        f"{row['transaction_pct']:.2f}% | "
        f"{row['fraud_rate_pct']:.2f}% |\n"
    )

report += f"""

---

## High-risk Transaction Monitoring

At the high-risk threshold of 0.70:

| Metric | Value |
|---|---:|
| High-risk transactions | {high_count} |
| High-risk share | {high_pct:.2f}% |
| Fraud transactions captured in high-risk group | {high_fraud_count} |
| High-risk precision | {high_precision:.2f}% |
| High-risk recall | {high_recall:.2f}% |

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
"""

report_path.write_text(report, encoding="utf-8")


# ============================================================
# 14. 保存完整 scored transactions
# ============================================================

df.to_csv(
    OUTPUT_DIR / "scored_transactions_with_risk_level.csv",
    index=False,
)


# ============================================================
# 15. 文件保存提示
# ============================================================

print("\n========== FILES SAVED ==========")
print("Scored transactions:", OUTPUT_DIR / "scored_transactions_with_risk_level.csv")
print("Risk level summary:", OUTPUT_DIR / "risk_level_summary.csv")
print("Action summary:", OUTPUT_DIR / "risk_action_summary.csv")
print("High risk transactions:", OUTPUT_DIR / "high_risk_transactions_model_based.csv")
print("Top 100 high risk:", OUTPUT_DIR / "top_100_high_risk_transactions.csv")
print("Risk scoring summary:", OUTPUT_DIR / "risk_scoring_summary_metrics.csv")
print("Business report:", report_path)
print("Figures saved to:", FIGURE_DIR)

print("\nRisk scoring completed successfully.")