from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib

# 终端运行时不弹图，直接保存图片
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
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
# 1. 读取 cleaned data
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\n========== DATA LOADED ==========")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# 2. 定义目标变量和特征变量
# ============================================================

TARGET_COL = "fraud_label"

# 这些列不用于建模
# transaction_id：交易编号，没有预测意义
# user_id：用户编号，直接放入模型容易让模型记住某些用户，不适合第一版 baseline
# timestamp：原始时间字符串不直接放入模型，我们使用已经拆出来的 month/hour/day_of_week 等
# risk_score：数据自带风险分数，可能和 fraud_label 强相关，避免 target leakage
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

print("\n========== MODELING SETUP ==========")
print("Target column:", TARGET_COL)
print("Number of features:", len(feature_cols))
print("Feature columns:")
print(feature_cols)

print("\nTarget distribution:")
print(y.value_counts())
print("\nTarget distribution percentage:")
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

# stratify=y 的意思是：
# 训练集和测试集里的 fraud 比例尽量保持一致
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\n========== TRAIN TEST SPLIT ==========")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train fraud rate:", y_train.mean() * 100)
print("y_test fraud rate :", y_test.mean() * 100)


# ============================================================
# 5. 预处理 Pipeline
# ============================================================

# 数值变量：
# 1. 如果有缺失值，用中位数填充
# 2. 用 StandardScaler 标准化
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# 类别变量：
# 1. 如果有缺失值，用 Unknown 填充
# 2. OneHotEncoder 把类别变量转成 0/1 哑变量
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# ColumnTransformer 会自动对不同类型的列做不同处理
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)


# ============================================================
# 6. Baseline 1：Majority Class Baseline
# ============================================================

# 最简单的 baseline：所有交易都预测成 0，也就是 Normal
# 这个模型很傻，但可以作为最低基准
y_pred_majority = np.zeros_like(y_test)

# 概率用训练集 fraud rate 作为常数概率
y_proba_majority = np.full(shape=len(y_test), fill_value=y_train.mean())

majority_metrics = {
    "model": "Majority Class Baseline",
    "accuracy": accuracy_score(y_test, y_pred_majority),
    "precision": precision_score(y_test, y_pred_majority, zero_division=0),
    "recall": recall_score(y_test, y_pred_majority, zero_division=0),
    "f1": f1_score(y_test, y_pred_majority, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba_majority),
    "pr_auc": average_precision_score(y_test, y_proba_majority),
}

print("\n========== MAJORITY CLASS BASELINE ==========")
for k, v in majority_metrics.items():
    if k == "model":
        print(k, ":", v)
    else:
        print(k, ":", round(v, 4))


# ============================================================
# 7. Baseline 2：Logistic Regression
# ============================================================

# Logistic Regression 是第一个真正的机器学习 baseline
# 它简单、可解释，经常作为分类任务的起点
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            ),
        ),
    ]
)

print("\n========== TRAINING LOGISTIC REGRESSION ==========")
logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)
y_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]

logistic_metrics = {
    "model": "Logistic Regression Baseline",
    "accuracy": accuracy_score(y_test, y_pred_logistic),
    "precision": precision_score(y_test, y_pred_logistic, zero_division=0),
    "recall": recall_score(y_test, y_pred_logistic, zero_division=0),
    "f1": f1_score(y_test, y_pred_logistic, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba_logistic),
    "pr_auc": average_precision_score(y_test, y_proba_logistic),
}

print("\n========== LOGISTIC REGRESSION BASELINE ==========")
for k, v in logistic_metrics.items():
    if k == "model":
        print(k, ":", v)
    else:
        print(k, ":", round(v, 4))


# ============================================================
# 8. 保存模型比较结果
# ============================================================

baseline_comparison = pd.DataFrame([majority_metrics, logistic_metrics])

# 转成百分比形式更方便看
metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
for col in metric_cols:
    baseline_comparison[col] = baseline_comparison[col] * 100

baseline_comparison = baseline_comparison.sort_values("pr_auc", ascending=False)

baseline_comparison.to_csv(
    OUTPUT_DIR / "baseline_model_comparison.csv",
    index=False,
)

print("\n========== BASELINE MODEL COMPARISON ==========")
print(baseline_comparison)


# ============================================================
# 9. 保存测试集预测结果
# ============================================================

predictions = X_test.copy()
predictions["actual_fraud_label"] = y_test.values
predictions["majority_pred"] = y_pred_majority
predictions["majority_fraud_probability"] = y_proba_majority
predictions["logistic_pred"] = y_pred_logistic
predictions["logistic_fraud_probability"] = y_proba_logistic

predictions.to_csv(
    OUTPUT_DIR / "baseline_model_predictions.csv",
    index=False,
)


# ============================================================
# 10. Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred_logistic)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"],
)

disp.plot(values_format="d")
plt.title("Confusion Matrix - Logistic Regression Baseline")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "14_confusion_matrix_logistic_baseline.png", dpi=160)
plt.close()


# ============================================================
# 11. ROC Curve
# ============================================================

fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_proba_logistic)
fpr_majority, tpr_majority, _ = roc_curve(y_test, y_proba_majority)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=f"Logistic Regression AUC = {logistic_metrics['roc_auc']:.3f}",
)
plt.plot(
    fpr_majority,
    tpr_majority,
    linestyle="--",
    label=f"Majority Baseline AUC = {majority_metrics['roc_auc']:.3f}",
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

plt.title("ROC Curve - Baseline Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Recall")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "15_roc_curve_baseline_models.png", dpi=160)
plt.close()


# ============================================================
# 12. Precision-Recall Curve
# ============================================================

precision_logistic, recall_logistic, _ = precision_recall_curve(y_test, y_proba_logistic)
precision_majority, recall_majority, _ = precision_recall_curve(y_test, y_proba_majority)

plt.figure(figsize=(7, 5))
plt.plot(
    recall_logistic,
    precision_logistic,
    label=f"Logistic Regression PR-AUC = {logistic_metrics['pr_auc']:.3f}",
)
plt.plot(
    recall_majority,
    precision_majority,
    linestyle="--",
    label=f"Majority Baseline PR-AUC = {majority_metrics['pr_auc']:.3f}",
)

plt.title("Precision-Recall Curve - Baseline Models")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "16_precision_recall_curve_baseline_models.png", dpi=160)
plt.close()


# ============================================================
# 13. 保存特征清单
# ============================================================

feature_info = pd.DataFrame(
    {
        "feature": feature_cols,
        "feature_type": [
            "categorical" if col in categorical_cols else "numeric"
            for col in feature_cols
        ],
    }
)

feature_info.to_csv(OUTPUT_DIR / "baseline_feature_list.csv", index=False)


# ============================================================
# 14. 总结
# ============================================================

print("\n========== FILES SAVED ==========")
print("Model comparison:", OUTPUT_DIR / "baseline_model_comparison.csv")
print("Predictions:", OUTPUT_DIR / "baseline_model_predictions.csv")
print("Feature list:", OUTPUT_DIR / "baseline_feature_list.csv")
print("Confusion matrix:", FIGURE_DIR / "14_confusion_matrix_logistic_baseline.png")
print("ROC curve:", FIGURE_DIR / "15_roc_curve_baseline_models.png")
print("Precision-Recall curve:", FIGURE_DIR / "16_precision_recall_curve_baseline_models.png")

print("\nBaseline modeling completed successfully.")