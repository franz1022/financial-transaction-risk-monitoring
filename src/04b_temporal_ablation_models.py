from __future__ import annotations

from pathlib import Path
import json
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# ============================================================
# 0. 路径与可修改参数
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "fraud_data_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "fraud_label"
TIMESTAMP_COL = "timestamp"

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

# 业务假设：人工审核能力约覆盖 20% 的交易。
# 阈值只在 Validation 上确定，然后原样应用到 Test。
TARGET_REVIEW_RATE = 0.20

RANDOM_STATE = 42

# 永远不进入模型的字段
NON_FEATURE_COLUMNS = {
    "transaction_id",
    "user_id",
    "timestamp",
    "fraud_label",
    "risk_score",  # 数据来源不透明，作为潜在代理字段排除
    "year",        # 当前数据只有 2023 年，为常量
}

# V2 的核心消融字段
SUSPECT_PROXY_COLUMN = "failed_transaction_count_7d"


# ============================================================
# 1. 工具函数
# ============================================================

def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns: " + ", ".join(missing)
        )


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = (
        df.sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
        .copy()
    )

    row_count = len(ordered)
    train_end = int(row_count * TRAIN_RATIO)
    validation_end = int(
        row_count * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_df = ordered.iloc[:train_end].copy()
    validation_df = ordered.iloc[train_end:validation_end].copy()
    test_df = ordered.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def build_preprocessor(
    X: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = (
        X.select_dtypes(include=["number", "bool"])
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_columns, categorical_columns


def threshold_for_review_capacity(
    probabilities: np.ndarray,
    target_review_rate: float,
) -> float:
    if not 0 < target_review_rate < 1:
        raise ValueError(
            "target_review_rate must be between 0 and 1."
        )

    quantile = 1 - target_review_rate
    return float(np.quantile(probabilities, quantile))


def safe_roc_auc(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    if y_true.nunique() < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "rows": int(len(y_true)),
        "fraud_count": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "review_rate": float(predictions.mean()),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": safe_roc_auc(
            y_true,
            probabilities,
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def build_models(
    positive_count: int,
    negative_count: int,
) -> dict[str, object]:
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBClassifier is not None:
        scale_pos_weight = (
            negative_count / positive_count
            if positive_count > 0
            else 1.0
        )

        models["XGBoost"] = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return models


def evaluate_rule_baseline(
    df: pd.DataFrame,
) -> dict[str, float | int | str]:
    y_true = df[TARGET_COL]
    probabilities = (
        df[SUSPECT_PROXY_COLUMN] >= 4
    ).astype(float).to_numpy()

    metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    metrics["model"] = "Rule Baseline"
    metrics["feature_set"] = (
        f"{SUSPECT_PROXY_COLUMN} >= 4"
    )

    return metrics


# ============================================================
# 2. 读取数据与时间划分
# ============================================================

print("=" * 72)
print("Temporal Feature Ablation Model Comparison")
print("=" * 72)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

require_columns(
    df,
    [
        TARGET_COL,
        TIMESTAMP_COL,
        SUSPECT_PROXY_COLUMN,
        "risk_score",
    ],
)

df[TIMESTAMP_COL] = pd.to_datetime(
    df[TIMESTAMP_COL],
    errors="coerce",
)

if df[TIMESTAMP_COL].isna().any():
    raise ValueError(
        "Invalid timestamps were found."
    )

if not set(df[TARGET_COL].dropna().unique()).issubset({0, 1}):
    raise ValueError(
        f"{TARGET_COL} must contain only 0 and 1."
    )

train_df, validation_df, test_df = temporal_split(df)

split_summary = pd.DataFrame(
    [
        {
            "split": "train",
            "rows": len(train_df),
            "start_timestamp": train_df[TIMESTAMP_COL].min(),
            "end_timestamp": train_df[TIMESTAMP_COL].max(),
            "fraud_count": int(train_df[TARGET_COL].sum()),
            "fraud_rate": float(train_df[TARGET_COL].mean()),
        },
        {
            "split": "validation",
            "rows": len(validation_df),
            "start_timestamp": validation_df[TIMESTAMP_COL].min(),
            "end_timestamp": validation_df[TIMESTAMP_COL].max(),
            "fraud_count": int(validation_df[TARGET_COL].sum()),
            "fraud_rate": float(validation_df[TARGET_COL].mean()),
        },
        {
            "split": "test",
            "rows": len(test_df),
            "start_timestamp": test_df[TIMESTAMP_COL].min(),
            "end_timestamp": test_df[TIMESTAMP_COL].max(),
            "fraud_count": int(test_df[TARGET_COL].sum()),
            "fraud_rate": float(test_df[TARGET_COL].mean()),
        },
    ]
)

split_summary.to_csv(
    OUTPUT_DIR / "temporal_model_split_summary.csv",
    index=False,
)

print()
print("Temporal split:")
print(split_summary.to_string(index=False))


# ============================================================
# 3. 定义两个特征集合
# ============================================================

all_candidate_features = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]

feature_sets = {
    # 仅作为敏感性基准，不允许成为最终部署候选。
    "benchmark_with_failed_count": all_candidate_features,
    # V2 的正式部署候选集合。
    "conservative_without_failed_count": [
        column
        for column in all_candidate_features
        if column != SUSPECT_PROXY_COLUMN
    ],
}

feature_manifest_rows = []

for feature_set_name, columns in feature_sets.items():
    for column in columns:
        feature_manifest_rows.append(
            {
                "feature_set": feature_set_name,
                "feature": column,
                "included": True,
            }
        )

feature_manifest = pd.DataFrame(feature_manifest_rows)

feature_manifest.to_csv(
    OUTPUT_DIR / "temporal_ablation_feature_manifest.csv",
    index=False,
)

print()
print("Feature set sizes:")
for name, columns in feature_sets.items():
    print(f"- {name}: {len(columns)} features")


# ============================================================
# 4. Validation 上比较模型与特征集合
# ============================================================

validation_rows: list[dict[str, float | int | str]] = []
trained_candidates: dict[tuple[str, str], Pipeline] = {}
candidate_thresholds: dict[tuple[str, str], float] = {}

y_train = train_df[TARGET_COL]
y_validation = validation_df[TARGET_COL]

positive_count = int(y_train.sum())
negative_count = int(len(y_train) - positive_count)

models = build_models(
    positive_count=positive_count,
    negative_count=negative_count,
)

for feature_set_name, features in feature_sets.items():
    X_train = train_df[features]
    X_validation = validation_df[features]

    preprocessor, numeric_columns, categorical_columns = (
        build_preprocessor(X_train)
    )

    for model_name, estimator in models.items():
        print()
        print(
            f"Training {model_name} "
            f"with {feature_set_name}..."
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(
                X_train,
                y_train,
            )

        validation_probabilities = pipeline.predict_proba(
            X_validation
        )[:, 1]

        threshold = threshold_for_review_capacity(
            validation_probabilities,
            TARGET_REVIEW_RATE,
        )

        metrics = calculate_metrics(
            y_validation,
            validation_probabilities,
            threshold,
        )

        metrics["feature_set"] = feature_set_name
        metrics["model"] = model_name
        metrics["numeric_feature_count"] = len(
            numeric_columns
        )
        metrics["categorical_feature_count"] = len(
            categorical_columns
        )
        metrics["selection_eligible"] = (
            feature_set_name
            == "conservative_without_failed_count"
        )

        validation_rows.append(metrics)

        key = (
            feature_set_name,
            model_name,
        )
        trained_candidates[key] = pipeline
        candidate_thresholds[key] = threshold


# 规则基线只作为诊断对照
rule_validation = evaluate_rule_baseline(
    validation_df
)
rule_validation["numeric_feature_count"] = 1
rule_validation["categorical_feature_count"] = 0
rule_validation["selection_eligible"] = False
validation_rows.append(rule_validation)

validation_results = pd.DataFrame(
    validation_rows
)

validation_results = validation_results.sort_values(
    [
        "selection_eligible",
        "pr_auc",
        "recall",
    ],
    ascending=[
        False,
        False,
        False,
    ],
)

validation_results.to_csv(
    OUTPUT_DIR / "temporal_ablation_validation_results.csv",
    index=False,
)

print()
print("Validation comparison:")
print(
    validation_results[
        [
            "feature_set",
            "model",
            "selection_eligible",
            "threshold",
            "review_rate",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
    ].to_string(index=False)
)


# ============================================================
# 5. 只在保守特征集合中选择正式模型
# ============================================================

eligible_results = validation_results.loc[
    validation_results["selection_eligible"]
].copy()

if eligible_results.empty:
    raise RuntimeError(
        "No eligible conservative models were trained."
    )

selected_row = (
    eligible_results.sort_values(
        [
            "pr_auc",
            "recall",
            "f1",
        ],
        ascending=False,
    )
    .iloc[0]
)

selected_feature_set = str(
    selected_row["feature_set"]
)
selected_model_name = str(
    selected_row["model"]
)

selected_key = (
    selected_feature_set,
    selected_model_name,
)

selected_pipeline = trained_candidates[selected_key]
selected_threshold = candidate_thresholds[selected_key]
selected_features = feature_sets[selected_feature_set]

selected_model_summary = pd.DataFrame(
    [
        {
            "selected_feature_set": selected_feature_set,
            "selected_model": selected_model_name,
            "selection_split": "validation",
            "selection_metric": "pr_auc",
            "validation_pr_auc": float(
                selected_row["pr_auc"]
            ),
            "validation_roc_auc": float(
                selected_row["roc_auc"]
            ),
            "validation_precision": float(
                selected_row["precision"]
            ),
            "validation_recall": float(
                selected_row["recall"]
            ),
            "validation_review_rate": float(
                selected_row["review_rate"]
            ),
            "selected_threshold": float(
                selected_threshold
            ),
            "target_review_rate": TARGET_REVIEW_RATE,
            "risk_score_excluded": True,
            "failed_count_excluded": True,
        }
    ]
)

selected_model_summary.to_csv(
    OUTPUT_DIR / "temporal_ablation_selected_model.csv",
    index=False,
)

print()
print("Selected model from Validation only:")
print(selected_model_summary.to_string(index=False))


# ============================================================
# 6. 独立 Test 只评估一次
# ============================================================

X_test = test_df[selected_features]
y_test = test_df[TARGET_COL]

test_probabilities = selected_pipeline.predict_proba(
    X_test
)[:, 1]

test_metrics = calculate_metrics(
    y_test,
    test_probabilities,
    selected_threshold,
)

test_metrics["feature_set"] = selected_feature_set
test_metrics["model"] = selected_model_name
test_metrics["threshold_source"] = "validation"
test_metrics["test_used_for_selection"] = False

final_test_results = pd.DataFrame(
    [test_metrics]
)

final_test_results.to_csv(
    OUTPUT_DIR / "temporal_ablation_final_test_metrics.csv",
    index=False,
)

# 规则基线的 Test 结果仅作背景对照
rule_test_results = pd.DataFrame(
    [
        evaluate_rule_baseline(
            test_df
        )
    ]
)

rule_test_results.to_csv(
    OUTPUT_DIR / "temporal_rule_baseline_test_metrics.csv",
    index=False,
)

print()
print("Independent Test result:")
print(
    final_test_results[
        [
            "feature_set",
            "model",
            "threshold",
            "review_rate",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "false_positive",
            "false_negative",
        ]
    ].to_string(index=False)
)

print()
print("Rule baseline Test result:")
print(
    rule_test_results[
        [
            "feature_set",
            "model",
            "review_rate",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "false_positive",
            "false_negative",
        ]
    ].to_string(index=False)
)


# ============================================================
# 7. 保存选择配置
# ============================================================

selection_config = {
    "selected_feature_set": selected_feature_set,
    "selected_model": selected_model_name,
    "selected_threshold": float(selected_threshold),
    "target_review_rate": TARGET_REVIEW_RATE,
    "selection_split": "validation",
    "final_evaluation_split": "test",
    "risk_score_excluded": True,
    "failed_transaction_count_7d_excluded": True,
    "features": selected_features,
}

with open(
    OUTPUT_DIR / "temporal_ablation_selection_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        selection_config,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 8. 绘图
# ============================================================

plot_validation = validation_results.loc[
    validation_results["model"] != "Rule Baseline"
].copy()

plot_validation["label"] = (
    plot_validation["model"]
    + "\n"
    + plot_validation["feature_set"]
)

plt.figure(figsize=(12, 7))

bars = plt.bar(
    plot_validation["label"],
    plot_validation["pr_auc"],
)

plt.title(
    "Validation PR-AUC: Full Benchmark vs Conservative Feature Set"
)
plt.xlabel("Model and feature set")
plt.ylabel("PR-AUC")
plt.xticks(rotation=35, ha="right")

for bar, value in zip(
    bars,
    plot_validation["pr_auc"],
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig(
    FIGURE_DIR
    / "37_temporal_ablation_validation_pr_auc.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()


test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title(
    "Independent Test Confusion Matrix\n"
    f"{selected_model_name}"
)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
plt.yticks([0, 1], ["Non-Fraud", "Fraud"])

for row_index in range(cm.shape[0]):
    for column_index in range(cm.shape[1]):
        plt.text(
            column_index,
            row_index,
            str(cm[row_index, column_index]),
            ha="center",
            va="center",
        )

plt.tight_layout()
plt.savefig(
    FIGURE_DIR
    / "38_temporal_ablation_test_confusion_matrix.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()


print()
print("Saved outputs:")
print(
    OUTPUT_DIR
    / "temporal_ablation_validation_results.csv"
)
print(
    OUTPUT_DIR
    / "temporal_ablation_selected_model.csv"
)
print(
    OUTPUT_DIR
    / "temporal_ablation_final_test_metrics.csv"
)
print(
    OUTPUT_DIR
    / "temporal_ablation_selection_config.json"
)
print()
print("Temporal ablation analysis completed successfully.")
