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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

from xgboost import XGBClassifier


# ============================================================
# 0. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "fraud_data_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 读取数据
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\n========== DATA LOADED ==========")
print("Shape:", df.shape)


# ============================================================
# 2. 定义目标变量和特征变量
# ============================================================

TARGET_COL = "fraud_label"

# 不用于建模的字段
# risk_score 不能用，因为它可能已经包含 fraud 信息，会造成 target leakage
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
print("Target:", TARGET_COL)
print("Number of features:", len(feature_cols))
print("Features:")
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

# stratify=y：让训练集和测试集保持相同 fraud 比例
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
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)


# ============================================================
# 6. 定义评估函数
# ============================================================

def evaluate_model(model_name, model, X_test, y_test):
    """
    对模型进行统一评估。
    返回 accuracy / precision / recall / f1 / roc_auc / pr_auc。
    """

    y_pred = model.predict(X_test)

    # 有些模型支持 predict_proba
    # fraud 概率取第二列 [:, 1]
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    return metrics, y_pred, y_proba


def print_metrics(metrics):
    print(f"\n========== {metrics['model']} ==========")
    for key, value in metrics.items():
        if key == "model":
            print(key, ":", value)
        else:
            print(key, ":", round(value, 4))


# ============================================================
# 7. 类别不平衡参数
# ============================================================

normal_count = (y_train == 0).sum()
fraud_count = (y_train == 1).sum()

# XGBoost 里常用 scale_pos_weight = negative / positive
scale_pos_weight = normal_count / fraud_count

print("\n========== CLASS IMBALANCE INFO ==========")
print("Normal count in train:", normal_count)
print("Fraud count in train :", fraud_count)
print("scale_pos_weight for XGBoost:", round(scale_pos_weight, 4))


# ============================================================
# 8. 定义模型
# ============================================================

models = {
    "Logistic Regression Balanced": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=10,
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),

    "Random Forest Balanced": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=10,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),

    "XGBoost Weighted": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    ),
}


# ============================================================
# 9. 训练和评估所有模型
# ============================================================

all_metrics = []
prediction_outputs = X_test.copy()
prediction_outputs["actual_fraud_label"] = y_test.values

roc_curve_data = {}
pr_curve_data = {}

for model_name, model in models.items():
    print(f"\nTraining model: {model_name}")
    model.fit(X_train, y_train)

    metrics, y_pred, y_proba = evaluate_model(
        model_name=model_name,
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    print_metrics(metrics)

    all_metrics.append(metrics)

    safe_model_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    prediction_outputs[f"{safe_model_name}_pred"] = y_pred
    prediction_outputs[f"{safe_model_name}_fraud_probability"] = y_proba

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curve_data[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": metrics["roc_auc"],
    }

    # PR curve data
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_curve_data[model_name] = {
        "precision": precision,
        "recall": recall,
        "pr_auc": metrics["pr_auc"],
    }

    # 每个模型保存一个 confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fraud"],
    )
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"17_confusion_matrix_{safe_model_name}.png",
        dpi=160,
    )
    plt.close()


# ============================================================
# 10. 保存模型比较结果
# ============================================================

model_comparison = pd.DataFrame(all_metrics)

metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

# 变成百分比，README 更好读
for col in metric_cols:
    model_comparison[col] = model_comparison[col] * 100

# 风控里 PR-AUC 和 Recall 很重要，这里按 PR-AUC 排名
model_comparison = model_comparison.sort_values(
    ["pr_auc", "recall"],
    ascending=False,
)

model_comparison.to_csv(
    OUTPUT_DIR / "class_imbalance_model_comparison.csv",
    index=False,
)

prediction_outputs.to_csv(
    OUTPUT_DIR / "class_imbalance_model_predictions.csv",
    index=False,
)

print("\n========== CLASS IMBALANCE MODEL COMPARISON ==========")
print(model_comparison)


# ============================================================
# 11. 模型比较柱状图
# ============================================================

plt.figure(figsize=(10, 5))
plt.bar(model_comparison["model"], model_comparison["pr_auc"])
plt.title("Model Comparison by PR-AUC")
plt.xlabel("Model")
plt.ylabel("PR-AUC (%)")
plt.xticks(rotation=25, ha="right")

for i, value in enumerate(model_comparison["pr_auc"]):
    plt.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "18_model_comparison_pr_auc.png", dpi=160)
plt.close()


plt.figure(figsize=(10, 5))
plt.bar(model_comparison["model"], model_comparison["recall"])
plt.title("Model Comparison by Recall")
plt.xlabel("Model")
plt.ylabel("Recall (%)")
plt.xticks(rotation=25, ha="right")

for i, value in enumerate(model_comparison["recall"]):
    plt.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "19_model_comparison_recall.png", dpi=160)
plt.close()


# ============================================================
# 12. ROC Curve 比较图
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, data in roc_curve_data.items():
    plt.plot(
        data["fpr"],
        data["tpr"],
        label=f"{model_name} AUC={data['roc_auc']:.3f}",
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.title("ROC Curve - Class Imbalance Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Recall")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "20_roc_curve_class_imbalance_models.png", dpi=160)
plt.close()


# ============================================================
# 13. Precision-Recall Curve 比较图
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, data in pr_curve_data.items():
    plt.plot(
        data["recall"],
        data["precision"],
        label=f"{model_name} PR-AUC={data['pr_auc']:.3f}",
    )

plt.title("Precision-Recall Curve - Class Imbalance Models")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "21_precision_recall_curve_class_imbalance_models.png", dpi=160)
plt.close()


# ============================================================
# 14. 提取 XGBoost 特征重要性
# ============================================================

# 先找到 XGBoost 模型
xgb_pipeline = models["XGBoost Weighted"]
xgb_classifier = xgb_pipeline.named_steps["classifier"]
xgb_preprocessor = xgb_pipeline.named_steps["preprocessor"]

try:
    feature_names = xgb_preprocessor.get_feature_names_out()
except Exception:
    feature_names = [f"feature_{i}" for i in range(len(xgb_classifier.feature_importances_))]

feature_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": xgb_classifier.feature_importances_,
    }
)

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False,
)

feature_importance.to_csv(
    OUTPUT_DIR / "xgboost_weighted_feature_importance.csv",
    index=False,
)

top_features = feature_importance.head(20)

plt.figure(figsize=(10, 7))
plt.barh(top_features["feature"][::-1], top_features["importance"][::-1])
plt.title("Top 20 XGBoost Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "22_xgboost_weighted_feature_importance.png", dpi=160)
plt.close()


# ============================================================
# 15. 总结最佳模型
# ============================================================

best_model_row = model_comparison.iloc[0]

summary_metrics = pd.DataFrame(
    [
        ["best_model_by_pr_auc", best_model_row["model"]],
        ["best_model_accuracy_pct", round(best_model_row["accuracy"], 2)],
        ["best_model_precision_pct", round(best_model_row["precision"], 2)],
        ["best_model_recall_pct", round(best_model_row["recall"], 2)],
        ["best_model_f1_pct", round(best_model_row["f1"], 2)],
        ["best_model_roc_auc_pct", round(best_model_row["roc_auc"], 2)],
        ["best_model_pr_auc_pct", round(best_model_row["pr_auc"], 2)],
        ["xgboost_scale_pos_weight", round(scale_pos_weight, 4)],
    ],
    columns=["metric", "value"],
)

summary_metrics.to_csv(
    OUTPUT_DIR / "class_imbalance_summary_metrics.csv",
    index=False,
)

print("\n========== BEST MODEL SUMMARY ==========")
print(summary_metrics)


# ============================================================
# 16. 保存文件提示
# ============================================================

print("\n========== FILES SAVED ==========")
print("Model comparison:", OUTPUT_DIR / "class_imbalance_model_comparison.csv")
print("Predictions:", OUTPUT_DIR / "class_imbalance_model_predictions.csv")
print("Summary metrics:", OUTPUT_DIR / "class_imbalance_summary_metrics.csv")
print("XGBoost feature importance:", OUTPUT_DIR / "xgboost_weighted_feature_importance.csv")
print("Figures saved to:", FIGURE_DIR)

print("\nClass imbalance modeling completed successfully.")