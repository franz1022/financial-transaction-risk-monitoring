from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib

# 终端运行，不弹出图片窗口，直接保存图
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


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
# 1. 读取上一阶段模型预测结果
# ============================================================

df = pd.read_csv(PREDICTION_FILE)

print("\n========== PREDICTION DATA LOADED ==========")
print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())


# ============================================================
# 2. 选择用于阈值分析的模型
# ============================================================

# 上一步最佳模型是 Random Forest
# 在 05 脚本里生成的概率列名是 random_forest_fraud_probability
TARGET_COL = "actual_fraud_label"
PROBA_COL = "random_forest_fraud_probability"

if TARGET_COL not in df.columns:
    raise ValueError(f"Cannot find target column: {TARGET_COL}")

if PROBA_COL not in df.columns:
    raise ValueError(f"Cannot find probability column: {PROBA_COL}")

y_true = df[TARGET_COL].values
y_proba = df[PROBA_COL].values

total_transactions = len(df)
total_fraud = int(np.sum(y_true == 1))
total_normal = int(np.sum(y_true == 0))
overall_fraud_rate = total_fraud / total_transactions * 100

print("\n========== BASIC INFO ==========")
print("Model probability column:", PROBA_COL)
print("Total test transactions:", total_transactions)
print("Total fraud transactions:", total_fraud)
print("Total normal transactions:", total_normal)
print(f"Overall fraud rate in test set: {overall_fraud_rate:.2f}%")


# ============================================================
# 3. 定义阈值评估函数
# ============================================================

def evaluate_threshold(threshold):
    """
    给定一个 threshold，计算风控指标。

    如果 fraud_probability >= threshold：
        预测为 Fraud，需要报警 / 人工审核

    如果 fraud_probability < threshold：
        预测为 Normal，自动通过
    """

    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    predicted_fraud_count = int(np.sum(y_pred == 1))
    predicted_normal_count = int(np.sum(y_pred == 0))

    review_rate = predicted_fraud_count / total_transactions
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    missed_fraud_rate = fn / total_fraud if total_fraud > 0 else 0

    return {
        "threshold": threshold,

        "predicted_fraud_count": predicted_fraud_count,
        "predicted_normal_count": predicted_normal_count,
        "review_rate": review_rate,

        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),

        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "missed_fraud_rate": missed_fraud_rate,
    }


# ============================================================
# 4. 对多个 threshold 做分析
# ============================================================

# 从 0.05 到 0.95，每隔 0.05 做一次分析
thresholds = np.arange(0.05, 1.00, 0.05)

threshold_results = []

for threshold in thresholds:
    result = evaluate_threshold(threshold)
    threshold_results.append(result)

threshold_df = pd.DataFrame(threshold_results)

# 转百分比，方便阅读
percentage_cols = [
    "review_rate",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "missed_fraud_rate",
]

threshold_df_pct = threshold_df.copy()

for col in percentage_cols:
    threshold_df_pct[col] = threshold_df_pct[col] * 100

threshold_df_pct.to_csv(
    OUTPUT_DIR / "threshold_analysis_random_forest.csv",
    index=False,
)

print("\n========== THRESHOLD ANALYSIS RESULT ==========")
print(threshold_df_pct)


# ============================================================
# 5. 找几个关键阈值
# ============================================================

# 1. F1 最大的阈值：precision 和 recall 综合最优
best_f1_row = threshold_df_pct.sort_values("f1", ascending=False).iloc[0]

# 2. Recall 至少 80% 的阈值：尽量多抓 fraud
high_recall_candidates = threshold_df_pct[threshold_df_pct["recall"] >= 80]

if len(high_recall_candidates) > 0:
    # 在 recall >= 80 的情况下，选择 precision 最高的
    high_recall_row = high_recall_candidates.sort_values(
        ["precision", "f1"],
        ascending=False,
    ).iloc[0]
else:
    high_recall_row = None

# 3. 审核量不超过 20% 的阈值：适合人工审核资源有限的情况
limited_review_candidates = threshold_df_pct[threshold_df_pct["review_rate"] <= 20]

if len(limited_review_candidates) > 0:
    # 在 review rate <= 20% 的情况下，选择 recall 最高的
    limited_review_row = limited_review_candidates.sort_values(
        ["recall", "precision"],
        ascending=False,
    ).iloc[0]
else:
    limited_review_row = None


summary_rows = []

summary_rows.append(
    {
        "strategy": "Best F1 Threshold",
        "threshold": best_f1_row["threshold"],
        "precision_pct": best_f1_row["precision"],
        "recall_pct": best_f1_row["recall"],
        "f1_pct": best_f1_row["f1"],
        "review_rate_pct": best_f1_row["review_rate"],
        "predicted_fraud_count": best_f1_row["predicted_fraud_count"],
        "false_positive": best_f1_row["false_positive"],
        "false_negative": best_f1_row["false_negative"],
        "business_meaning": "Balance precision and recall",
    }
)

if high_recall_row is not None:
    summary_rows.append(
        {
            "strategy": "High Recall Threshold",
            "threshold": high_recall_row["threshold"],
            "precision_pct": high_recall_row["precision"],
            "recall_pct": high_recall_row["recall"],
            "f1_pct": high_recall_row["f1"],
            "review_rate_pct": high_recall_row["review_rate"],
            "predicted_fraud_count": high_recall_row["predicted_fraud_count"],
            "false_positive": high_recall_row["false_positive"],
            "false_negative": high_recall_row["false_negative"],
            "business_meaning": "Catch more fraud but create more reviews",
        }
    )

if limited_review_row is not None:
    summary_rows.append(
        {
            "strategy": "Limited Review Capacity Threshold",
            "threshold": limited_review_row["threshold"],
            "precision_pct": limited_review_row["precision"],
            "recall_pct": limited_review_row["recall"],
            "f1_pct": limited_review_row["f1"],
            "review_rate_pct": limited_review_row["review_rate"],
            "predicted_fraud_count": limited_review_row["predicted_fraud_count"],
            "false_positive": limited_review_row["false_positive"],
            "false_negative": limited_review_row["false_negative"],
            "business_meaning": "Control manual review workload",
        }
    )

threshold_recommendations = pd.DataFrame(summary_rows)

threshold_recommendations.to_csv(
    OUTPUT_DIR / "threshold_strategy_recommendations.csv",
    index=False,
)

print("\n========== THRESHOLD STRATEGY RECOMMENDATIONS ==========")
print(threshold_recommendations)


# ============================================================
# 6. 画图 1：Precision / Recall / F1 随 threshold 变化
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df_pct["threshold"],
    threshold_df_pct["precision"],
    marker="o",
    label="Precision",
)

plt.plot(
    threshold_df_pct["threshold"],
    threshold_df_pct["recall"],
    marker="o",
    label="Recall",
)

plt.plot(
    threshold_df_pct["threshold"],
    threshold_df_pct["f1"],
    marker="o",
    label="F1-score",
)

plt.title("Precision, Recall and F1-score by Threshold")
plt.xlabel("Fraud Probability Threshold")
plt.ylabel("Score (%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "23_precision_recall_f1_by_threshold.png", dpi=160)
plt.close()


# ============================================================
# 7. 画图 2：Review Rate 随 threshold 变化
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df_pct["threshold"],
    threshold_df_pct["review_rate"],
    marker="o",
)

plt.title("Manual Review Rate by Threshold")
plt.xlabel("Fraud Probability Threshold")
plt.ylabel("Transactions Flagged for Review (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "24_review_rate_by_threshold.png", dpi=160)
plt.close()


# ============================================================
# 8. 画图 3：False Positive / False Negative 数量变化
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df_pct["threshold"],
    threshold_df_pct["false_positive"],
    marker="o",
    label="False Positives",
)

plt.plot(
    threshold_df_pct["threshold"],
    threshold_df_pct["false_negative"],
    marker="o",
    label="False Negatives",
)

plt.title("False Positives and False Negatives by Threshold")
plt.xlabel("Fraud Probability Threshold")
plt.ylabel("Number of Transactions")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "25_false_positive_false_negative_by_threshold.png", dpi=160)
plt.close()


# ============================================================
# 9. 画图 4：几个关键阈值的 Confusion Matrix
# ============================================================

selected_thresholds = [0.3, 0.5, 0.7]

for threshold in selected_thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fraud"],
    )

    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - Random Forest Threshold {threshold:.2f}")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"26_confusion_matrix_random_forest_threshold_{threshold:.2f}.png",
        dpi=160,
    )
    plt.close()


# ============================================================
# 10. 生成一份业务解释 Markdown
# ============================================================

business_report_path = OUTPUT_DIR / "threshold_analysis_business_report.md"

best_f1_threshold = best_f1_row["threshold"]
best_f1_precision = best_f1_row["precision"]
best_f1_recall = best_f1_row["recall"]
best_f1_review_rate = best_f1_row["review_rate"]

report = f"""# Threshold Analysis Business Report

## Objective

The goal of this analysis is to evaluate how different fraud probability thresholds affect fraud detection performance and manual review workload.

In fraud monitoring, the model does not only produce a 0/1 prediction. It also produces a fraud probability. By changing the decision threshold, the business team can balance two competing goals:

- catching more fraudulent transactions
- controlling false alarms and manual review workload

---

## Key Dataset Information

| Metric | Value |
|---|---:|
| Test transactions | {total_transactions} |
| Fraud transactions | {total_fraud} |
| Normal transactions | {total_normal} |
| Overall fraud rate | {overall_fraud_rate:.2f}% |

---

## Recommended Threshold Based on F1-score

| Metric | Value |
|---|---:|
| Recommended threshold | {best_f1_threshold:.2f} |
| Precision | {best_f1_precision:.2f}% |
| Recall | {best_f1_recall:.2f}% |
| F1-score | {best_f1_row["f1"]:.2f}% |
| Manual review rate | {best_f1_review_rate:.2f}% |
| Transactions flagged for review | {int(best_f1_row["predicted_fraud_count"])} |
| False positives | {int(best_f1_row["false_positive"])} |
| False negatives | {int(best_f1_row["false_negative"])} |

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
"""

business_report_path.write_text(report, encoding="utf-8")


# ============================================================
# 11. 文件保存提示
# ============================================================

print("\n========== FILES SAVED ==========")
print("Threshold analysis:", OUTPUT_DIR / "threshold_analysis_random_forest.csv")
print("Threshold recommendations:", OUTPUT_DIR / "threshold_strategy_recommendations.csv")
print("Business report:", business_report_path)
print("Figures saved to:", FIGURE_DIR)

print("\nThreshold analysis completed successfully.")