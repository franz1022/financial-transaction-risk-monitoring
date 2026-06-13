from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib

# 终端运行，不弹出图片窗口，直接保存图
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


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
# 1. 读取数据
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\n========== DATA LOADED ==========")
print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())


# ============================================================
# 2. 定义目标变量和特征变量
# ============================================================

TARGET_COL = "fraud_label"

# 注意：
# 无监督异常检测训练时不使用 fraud_label。
# fraud_label 只在最后用来评估 anomaly detection 找到的异常交易里面有多少是真的 fraud。
#
# risk_score 仍然排除，因为它可能已经包含 fraud 信息，会造成 target leakage。
DROP_COLS = [
    "transaction_id",
    "user_id",
    "timestamp",
    "risk_score",
    TARGET_COL,
]

feature_cols = [col for col in df.columns if col not in DROP_COLS]

X = df[feature_cols]
y = df[TARGET_COL]

print("\n========== ANOMALY DETECTION SETUP ==========")
print("Number of features:", len(feature_cols))
print("Feature columns:")
print(feature_cols)

print("\nFraud label distribution, used only for evaluation:")
print(y.value_counts())
print(y.value_counts(normalize=True) * 100)


# ============================================================
# 3. 区分类别变量和数值变量
# ============================================================

categorical_cols = X.select_dtypes(include=["object", "category", "str"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

print("\n========== FEATURE TYPES ==========")
print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)


# ============================================================
# 4. Train-test split
# ============================================================

# 这里 split 主要是为了和前面监督学习模型保持一致
# 但 Isolation Forest 训练时不会用 y_train
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\n========== TRAIN TEST SPLIT ==========")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train fraud rate:", round(y_train.mean() * 100, 2))
print("y_test fraud rate :", round(y_test.mean() * 100, 2))


# ============================================================
# 5. 预处理 Pipeline
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # 输出 dense array，Isolation Forest 使用更稳定
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)


# ============================================================
# 6. 训练 Isolation Forest
# ============================================================

# contamination 是模型预期异常比例。
# 真实风控里 fraud rate 通常未知，所以这里先设成 10%，表示先找最异常的一批交易。
isolation_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            IsolationForest(
                n_estimators=200,
                contamination=0.10,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("\n========== TRAINING ISOLATION FOREST ==========")
isolation_forest.fit(X_train)

print("Isolation Forest training completed.")


# ============================================================
# 7. 生成 anomaly score
# ============================================================

# IsolationForest.decision_function:
# 分数越高越正常，越低越异常。
#
# 所以我们取负数：
# anomaly_score 越高 = 越异常
decision_scores = isolation_forest.decision_function(X_test)
anomaly_scores = -decision_scores

# 为了更像风险分数，我们把 anomaly_score 转成 0-100 percentile score
# 排名越靠前，异常分越高
anomaly_score_percentile = pd.Series(anomaly_scores).rank(pct=True).values * 100

scored_df = X_test.copy()
scored_df["actual_fraud_label"] = y_test.values
scored_df["isolation_forest_anomaly_score"] = anomaly_scores
scored_df["isolation_forest_anomaly_score_pct"] = anomaly_score_percentile

print("\n========== ANOMALY SCORE SUMMARY ==========")
print(scored_df["isolation_forest_anomaly_score_pct"].describe())


# ============================================================
# 8. 用不同 review rate 评估异常检测效果
# ============================================================

# 这里的 review_rate 可以理解为：
# 如果业务只愿意人工检查 top 5% / 10% / 20% 最异常交易，效果如何？
review_rates = [0.05, 0.10, 0.15, 0.20, 0.30]

results = []

for review_rate in review_rates:
    # top review_rate 的 anomaly score 作为异常交易
    threshold = np.quantile(anomaly_scores, 1 - review_rate)

    y_pred_anomaly = (anomaly_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_anomaly).ravel()

    precision = precision_score(y_test, y_pred_anomaly, zero_division=0)
    recall = recall_score(y_test, y_pred_anomaly, zero_division=0)
    f1 = f1_score(y_test, y_pred_anomaly, zero_division=0)

    predicted_anomaly_count = int(y_pred_anomaly.sum())
    fraud_captured = int(tp)
    false_alerts = int(fp)
    missed_fraud = int(fn)

    results.append(
        {
            "review_rate": review_rate,
            "review_rate_pct": review_rate * 100,
            "anomaly_threshold": threshold,
            "predicted_anomaly_count": predicted_anomaly_count,
            "fraud_captured": fraud_captured,
            "false_alerts": false_alerts,
            "missed_fraud": missed_fraud,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

anomaly_eval = pd.DataFrame(results)

# 转百分比便于阅读
for col in ["precision", "recall", "f1"]:
    anomaly_eval[col + "_pct"] = anomaly_eval[col] * 100

anomaly_eval.to_csv(
    OUTPUT_DIR / "anomaly_detection_review_rate_evaluation.csv",
    index=False,
)

print("\n========== ANOMALY DETECTION REVIEW RATE EVALUATION ==========")
print(
    anomaly_eval[
        [
            "review_rate_pct",
            "predicted_anomaly_count",
            "fraud_captured",
            "false_alerts",
            "missed_fraud",
            "precision_pct",
            "recall_pct",
            "f1_pct",
        ]
    ]
)


# ============================================================
# 9. 连续 anomaly score 的整体评估
# ============================================================

# 这里不是把 anomaly detection 当成严格分类器，
# 而是看 anomaly_score 是否能把 fraud 排到更靠前的位置。
roc_auc = roc_auc_score(y_test, anomaly_scores)
pr_auc = average_precision_score(y_test, anomaly_scores)

print("\n========== CONTINUOUS ANOMALY SCORE EVALUATION ==========")
print(f"ROC-AUC based on anomaly score: {roc_auc:.4f}")
print(f"PR-AUC based on anomaly score : {pr_auc:.4f}")


# ============================================================
# 10. 保存 scored transactions
# ============================================================

# 默认标记 top 10% 为 anomaly，方便后续 README / dashboard 展示
default_review_rate = 0.10
default_threshold = np.quantile(anomaly_scores, 1 - default_review_rate)

scored_df["is_anomaly_top_10pct"] = (
    scored_df["isolation_forest_anomaly_score"] >= default_threshold
).astype(int)

scored_df.to_csv(
    OUTPUT_DIR / "anomaly_detection_scored_transactions.csv",
    index=False,
)


# ============================================================
# 11. 图 1：Anomaly Score 分布
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(
    scored_df.loc[scored_df["actual_fraud_label"] == 0, "isolation_forest_anomaly_score_pct"],
    bins=40,
    alpha=0.6,
    label="Normal",
    density=True,
)

plt.hist(
    scored_df.loc[scored_df["actual_fraud_label"] == 1, "isolation_forest_anomaly_score_pct"],
    bins=40,
    alpha=0.6,
    label="Fraud",
    density=True,
)

plt.title("Isolation Forest Anomaly Score Distribution")
plt.xlabel("Anomaly Score Percentile")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "31_isolation_forest_anomaly_score_distribution.png", dpi=160)
plt.close()


# ============================================================
# 12. 图 2：Review Rate 下 Precision / Recall / F1
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    anomaly_eval["review_rate_pct"],
    anomaly_eval["precision_pct"],
    marker="o",
    label="Precision",
)

plt.plot(
    anomaly_eval["review_rate_pct"],
    anomaly_eval["recall_pct"],
    marker="o",
    label="Recall",
)

plt.plot(
    anomaly_eval["review_rate_pct"],
    anomaly_eval["f1_pct"],
    marker="o",
    label="F1-score",
)

plt.title("Anomaly Detection Performance by Review Rate")
plt.xlabel("Top Anomaly Transactions Reviewed (%)")
plt.ylabel("Score (%)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "32_anomaly_detection_performance_by_review_rate.png", dpi=160)
plt.close()


# ============================================================
# 13. 图 3：Review Rate 下 fraud captured / false alerts
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    anomaly_eval["review_rate_pct"],
    anomaly_eval["fraud_captured"],
    marker="o",
    label="Fraud Captured",
)

plt.plot(
    anomaly_eval["review_rate_pct"],
    anomaly_eval["false_alerts"],
    marker="o",
    label="False Alerts",
)

plt.title("Fraud Captured and False Alerts by Review Rate")
plt.xlabel("Top Anomaly Transactions Reviewed (%)")
plt.ylabel("Number of Transactions")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "33_fraud_captured_false_alerts_by_review_rate.png", dpi=160)
plt.close()


# ============================================================
# 14. 图 4：Top 10% anomaly 的 Confusion Matrix
# ============================================================

y_pred_top10 = scored_df["is_anomaly_top_10pct"].values
cm = confusion_matrix(y_test, y_pred_top10)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"],
)

disp.plot(values_format="d")
plt.title("Confusion Matrix - Isolation Forest Top 10% Anomalies")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "34_confusion_matrix_isolation_forest_top10pct.png", dpi=160)
plt.close()


# ============================================================
# 15. 和监督模型结果做简单比较
# ============================================================

comparison_rows = [
    {
        "model_type": "Unsupervised Anomaly Detection",
        "model": "Isolation Forest",
        "roc_auc_pct": roc_auc * 100,
        "pr_auc_pct": pr_auc * 100,
        "business_use_case": "Detect unusual transactions when fraud labels are incomplete or delayed",
    }
]

# 如果前面监督学习结果存在，就读进来做对比
supervised_file = OUTPUT_DIR / "class_imbalance_summary_metrics.csv"

if supervised_file.exists():
    supervised_summary = pd.read_csv(supervised_file)

    def get_metric(metric_name):
        row = supervised_summary[supervised_summary["metric"] == metric_name]
        if len(row) == 0:
            return None
        return row["value"].iloc[0]

    best_model = get_metric("best_model_by_pr_auc")
    best_roc_auc = get_metric("best_model_roc_auc_pct")
    best_pr_auc = get_metric("best_model_pr_auc_pct")

    comparison_rows.append(
        {
            "model_type": "Supervised Classification",
            "model": best_model,
            "roc_auc_pct": best_roc_auc,
            "pr_auc_pct": best_pr_auc,
            "business_use_case": "Predict fraud when historical fraud labels are available",
        }
    )

model_type_comparison = pd.DataFrame(comparison_rows)

model_type_comparison.to_csv(
    OUTPUT_DIR / "supervised_vs_anomaly_detection_comparison.csv",
    index=False,
)

print("\n========== SUPERVISED VS ANOMALY DETECTION COMPARISON ==========")
print(model_type_comparison)


# ============================================================
# 16. 生成业务报告
# ============================================================

best_review_row = anomaly_eval.sort_values("f1", ascending=False).iloc[0]

report_path = OUTPUT_DIR / "anomaly_detection_business_report.md"

report = f"""# Anomaly Detection Business Report

## Objective

This step adds an unsupervised anomaly detection layer to the transaction risk monitoring system.

Unlike supervised fraud classification models, Isolation Forest does not use fraud labels during training. It learns the general pattern of normal transaction behavior and assigns higher anomaly scores to unusual transactions.

This is useful when:

- fraud labels are incomplete
- fraud labels are delayed
- new fraud patterns have not yet been labeled
- the business wants an early warning system for unusual transactions

---

## Model

| Item | Value |
|---|---|
| Model | Isolation Forest |
| Training mode | Unsupervised |
| Features used | Transaction, device, location, merchant, behavior, and time features |
| Excluded columns | transaction_id, user_id, timestamp, risk_score, fraud_label |

The existing risk_score was excluded to avoid target leakage.

---

## Continuous Anomaly Score Performance

| Metric | Value |
|---|---:|
| ROC-AUC | {roc_auc * 100:.2f}% |
| PR-AUC | {pr_auc * 100:.2f}% |

---

## Best Review-rate Strategy by F1-score

| Metric | Value |
|---|---:|
| Review rate | {best_review_row["review_rate_pct"]:.2f}% |
| Transactions reviewed | {int(best_review_row["predicted_anomaly_count"])} |
| Fraud captured | {int(best_review_row["fraud_captured"])} |
| False alerts | {int(best_review_row["false_alerts"])} |
| Precision | {best_review_row["precision_pct"]:.2f}% |
| Recall | {best_review_row["recall_pct"]:.2f}% |
| F1-score | {best_review_row["f1_pct"]:.2f}% |

---

## Business Interpretation

Isolation Forest is not expected to replace supervised fraud models when reliable fraud labels are available.  
Instead, it works as a complementary early-warning layer for suspicious and unusual transaction patterns.

The business can choose a review rate, such as top 5%, 10%, or 20% most anomalous transactions, depending on manual review capacity.

---

## Recommended Usage

- Use supervised models for known fraud pattern detection.
- Use Isolation Forest to detect unusual transactions and potential new fraud patterns.
- Combine anomaly scores with model-based risk scores for a multi-layer risk monitoring system.
"""

report_path.write_text(report, encoding="utf-8")


# ============================================================
# 17. 保存总结指标
# ============================================================

summary_metrics = pd.DataFrame(
    [
        ["model", "Isolation Forest"],
        ["training_mode", "Unsupervised"],
        ["test_transactions", len(X_test)],
        ["test_fraud_transactions", int(y_test.sum())],
        ["roc_auc_pct", round(roc_auc * 100, 2)],
        ["pr_auc_pct", round(pr_auc * 100, 2)],
        ["default_review_rate_pct", default_review_rate * 100],
        ["default_top10_anomaly_threshold", round(default_threshold, 6)],
        ["best_review_rate_by_f1_pct", round(best_review_row["review_rate_pct"], 2)],
        ["best_precision_pct", round(best_review_row["precision_pct"], 2)],
        ["best_recall_pct", round(best_review_row["recall_pct"], 2)],
        ["best_f1_pct", round(best_review_row["f1_pct"], 2)],
    ],
    columns=["metric", "value"],
)

summary_metrics.to_csv(
    OUTPUT_DIR / "anomaly_detection_summary_metrics.csv",
    index=False,
)

print("\n========== ANOMALY DETECTION SUMMARY METRICS ==========")
print(summary_metrics)


# ============================================================
# 18. 文件保存提示
# ============================================================

print("\n========== FILES SAVED ==========")
print("Anomaly evaluation:", OUTPUT_DIR / "anomaly_detection_review_rate_evaluation.csv")
print("Scored transactions:", OUTPUT_DIR / "anomaly_detection_scored_transactions.csv")
print("Summary metrics:", OUTPUT_DIR / "anomaly_detection_summary_metrics.csv")
print("Business report:", report_path)
print("Supervised vs anomaly comparison:", OUTPUT_DIR / "supervised_vs_anomaly_detection_comparison.csv")
print("Figures saved to:", FIGURE_DIR)

print("\nAnomaly detection completed successfully.")