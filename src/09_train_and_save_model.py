from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# 0. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "fraud_data_cleaned.csv"

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "fraud_detection_random_forest_pipeline.pkl"
METADATA_FILE = MODELS_DIR / "model_metadata.json"
FEATURE_FILE = MODELS_DIR / "model_feature_columns.json"
SAMPLE_INPUT_FILE = OUTPUT_DIR / "sample_transaction_input.json"


# ============================================================
# 1. 读取 cleaned data
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

# 不用于模型训练的字段：
# transaction_id：交易编号，没有泛化预测意义
# user_id：用户编号，避免模型记住用户
# timestamp：原始字符串不直接进入模型，使用拆出来的 year/month/day/hour/day_of_week
# risk_score：原数据自带风险分数，可能造成 target leakage
# fraud_label：目标变量，不能放进 X
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
# 4. Train-test split，用于最后确认模型效果
# ============================================================

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
# 6. 定义最终 Random Forest 模型
# ============================================================

final_model = Pipeline(
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
)


# ============================================================
# 7. 先在训练集训练，并在测试集评估
# ============================================================

print("\n========== TRAINING FINAL RANDOM FOREST FOR EVALUATION ==========")

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

evaluation_metrics = {
    "model": "Random Forest",
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "pr_auc": average_precision_score(y_test, y_proba),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
}

print("\n========== FINAL MODEL EVALUATION ==========")
for key, value in evaluation_metrics.items():
    if isinstance(value, float):
        print(key, ":", round(value, 4))
    else:
        print(key, ":", value)


# ============================================================
# 8. 用全量数据重新训练最终模型
# ============================================================

# 解释：
# 前面 train-test split 是为了评估模型效果。
# 确认模型效果后，保存部署模型时，可以用全量数据重新训练，
# 让模型利用所有历史样本。
print("\n========== RETRAINING FINAL MODEL ON FULL DATA ==========")

deployment_model = Pipeline(
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
)

deployment_model.fit(X, y)

print("Deployment model trained on full dataset.")


# ============================================================
# 9. 保存模型
# ============================================================

joblib.dump(deployment_model, MODEL_FILE)

print("\n========== MODEL SAVED ==========")
print("Model saved to:", MODEL_FILE)


# ============================================================
# 10. 保存 feature columns
# ============================================================

feature_info = {
    "target_col": TARGET_COL,
    "feature_cols": feature_cols,
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "excluded_cols": DROP_COLS,
}

FEATURE_FILE.write_text(
    json.dumps(feature_info, indent=4),
    encoding="utf-8",
)

print("Feature columns saved to:", FEATURE_FILE)


# ============================================================
# 11. 保存模型 metadata
# ============================================================

metadata = {
    "project_name": "Financial Transaction Risk Monitoring",
    "model_name": "Random Forest Fraud Detection Pipeline",
    "model_file": str(MODEL_FILE),
    "training_data": str(DATA_FILE),
    "number_of_rows": int(df.shape[0]),
    "number_of_features": len(feature_cols),
    "target_column": TARGET_COL,
    "risk_score_excluded": True,
    "risk_score_exclusion_reason": "Existing risk_score was excluded to avoid potential target leakage.",
    "low_risk_threshold": 0.20,
    "high_risk_threshold": 0.70,
    "evaluation_metrics": evaluation_metrics,
    "recommended_actions": {
        "Low": "Auto Approve",
        "Medium": "Monitor / Secondary Check",
        "High": "Manual Review / Alert",
    },
}

METADATA_FILE.write_text(
    json.dumps(metadata, indent=4),
    encoding="utf-8",
)

print("Model metadata saved to:", METADATA_FILE)


# ============================================================
# 12. 保存一个 sample input，方便后面 FastAPI 测试
# ============================================================

sample_row = X.iloc[0].copy()

# 把 pandas/numpy 类型转成普通 Python 类型，方便保存成 JSON
sample_input = {}

for col in feature_cols:
    value = sample_row[col]

    if pd.isna(value):
        sample_input[col] = None
    elif hasattr(value, "item"):
        sample_input[col] = value.item()
    else:
        sample_input[col] = value

SAMPLE_INPUT_FILE.write_text(
    json.dumps(sample_input, indent=4),
    encoding="utf-8",
)

print("Sample input saved to:", SAMPLE_INPUT_FILE)


# ============================================================
# 13. 保存 evaluation summary
# ============================================================

evaluation_summary = pd.DataFrame(
    [
        ["model", evaluation_metrics["model"]],
        ["accuracy_pct", round(evaluation_metrics["accuracy"] * 100, 2)],
        ["precision_pct", round(evaluation_metrics["precision"] * 100, 2)],
        ["recall_pct", round(evaluation_metrics["recall"] * 100, 2)],
        ["f1_pct", round(evaluation_metrics["f1"] * 100, 2)],
        ["roc_auc_pct", round(evaluation_metrics["roc_auc"] * 100, 2)],
        ["pr_auc_pct", round(evaluation_metrics["pr_auc"] * 100, 2)],
        ["true_negative", evaluation_metrics["true_negative"]],
        ["false_positive", evaluation_metrics["false_positive"]],
        ["false_negative", evaluation_metrics["false_negative"]],
        ["true_positive", evaluation_metrics["true_positive"]],
        ["model_saved_to", str(MODEL_FILE)],
    ],
    columns=["metric", "value"],
)

evaluation_summary.to_csv(
    OUTPUT_DIR / "final_saved_model_evaluation_summary.csv",
    index=False,
)

print("\n========== FILES SAVED ==========")
print("Model:", MODEL_FILE)
print("Metadata:", METADATA_FILE)
print("Feature columns:", FEATURE_FILE)
print("Sample input:", SAMPLE_INPUT_FILE)
print("Evaluation summary:", OUTPUT_DIR / "final_saved_model_evaluation_summary.csv")

print("\nFinal model training and saving completed successfully.")