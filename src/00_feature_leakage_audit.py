from __future__ import annotations

from pathlib import Path

import matplotlib

# 终端运行时不弹出绘图窗口
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
USER_COL = "user_id"

FAILED_COUNT_COL = "failed_transaction_count_7d"
RISK_SCORE_COL = "risk_score"
PREVIOUS_FRAUD_COL = "previous_fraudulent_activity"
IP_FLAG_COL = "ip_address_flag"

# 使用时间顺序的 60% / 20% / 20% 划分，仅用于审计，不会训练最终模型
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

RULE_THRESHOLD = 4


# ============================================================
# 1. 工具函数
# ============================================================

def require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """检查必要字段是否存在。"""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def safe_roc_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """只有目标包含两个类别时才计算 ROC-AUC。"""
    if y_true.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series,
) -> dict[str, float | int]:
    """统一计算规则或模型的分类指标。"""
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    review_rate = float(y_pred.mean())
    fraud_rate = float(y_true.mean())

    return {
        "rows": int(len(y_true)),
        "fraud_count": int(y_true.sum()),
        "fraud_rate": fraud_rate,
        "predicted_positive_count": int(y_pred.sum()),
        "review_rate": review_rate,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_score,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def assign_temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """按时间顺序给数据加上 Train / Validation / Test 标记。"""
    sorted_df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True).copy()

    row_count = len(sorted_df)
    train_end = int(row_count * TRAIN_RATIO)
    validation_end = int(
        row_count * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    split = np.full(row_count, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:validation_end] = "validation"

    sorted_df["temporal_split"] = split
    return sorted_df


def build_group_fraud_summary(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """按指定字段汇总样本数、欺诈数与欺诈率。"""
    summary = (
        df.groupby(
            group_column,
            dropna=False,
        )[TARGET_COL]
        .agg(
            transaction_count="size",
            fraud_count="sum",
            fraud_rate="mean",
        )
        .reset_index()
        .sort_values(group_column)
    )

    summary["fraud_rate_pct"] = summary["fraud_rate"] * 100
    return summary


def audit_numeric_feature(
    df: pd.DataFrame,
    feature: str,
) -> dict[str, float | int | str]:
    """检查数值特征与目标变量的简单关系。"""
    valid = df[[feature, TARGET_COL]].dropna()

    non_fraud = valid.loc[
        valid[TARGET_COL] == 0,
        feature,
    ]
    fraud = valid.loc[
        valid[TARGET_COL] == 1,
        feature,
    ]

    correlation = valid[feature].corr(valid[TARGET_COL])

    return {
        "feature": feature,
        "feature_type": "numeric",
        "non_null_rows": int(len(valid)),
        "unique_values": int(valid[feature].nunique()),
        "non_fraud_mean": float(non_fraud.mean()),
        "fraud_mean": float(fraud.mean()),
        "mean_difference": float(
            fraud.mean() - non_fraud.mean()
        ),
        "target_correlation": (
            float(correlation)
            if pd.notna(correlation)
            else np.nan
        ),
        "minimum_group_fraud_rate": np.nan,
        "maximum_group_fraud_rate": np.nan,
        "fraud_rate_range": np.nan,
    }


def audit_categorical_feature(
    df: pd.DataFrame,
    feature: str,
) -> dict[str, float | int | str]:
    """检查类别特征不同取值之间的欺诈率差异。"""
    grouped = (
        df.groupby(
            feature,
            dropna=False,
        )[TARGET_COL]
        .agg(
            transaction_count="size",
            fraud_rate="mean",
        )
        .reset_index()
    )

    return {
        "feature": feature,
        "feature_type": "categorical",
        "non_null_rows": int(df[feature].notna().sum()),
        "unique_values": int(df[feature].nunique(dropna=False)),
        "non_fraud_mean": np.nan,
        "fraud_mean": np.nan,
        "mean_difference": np.nan,
        "target_correlation": np.nan,
        "minimum_group_fraud_rate": float(
            grouped["fraud_rate"].min()
        ),
        "maximum_group_fraud_rate": float(
            grouped["fraud_rate"].max()
        ),
        "fraud_rate_range": float(
            grouped["fraud_rate"].max()
            - grouped["fraud_rate"].min()
        ),
    }


# ============================================================
# 2. 读取与验证数据
# ============================================================

print("=" * 70)
print("Financial Transaction Feature Leakage Audit")
print("=" * 70)

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
        USER_COL,
        FAILED_COUNT_COL,
        RISK_SCORE_COL,
        PREVIOUS_FRAUD_COL,
        IP_FLAG_COL,
    ],
)

df[TIMESTAMP_COL] = pd.to_datetime(
    df[TIMESTAMP_COL],
    errors="coerce",
)

if df[TIMESTAMP_COL].isna().any():
    invalid_count = int(df[TIMESTAMP_COL].isna().sum())
    raise ValueError(
        f"{invalid_count} rows contain invalid timestamps."
    )

if not set(df[TARGET_COL].dropna().unique()).issubset({0, 1}):
    raise ValueError(
        f"{TARGET_COL} must contain only 0 and 1."
    )

df = assign_temporal_split(df)

print(f"Rows: {len(df):,}")
print(f"Columns: {df.shape[1]:,}")
print(
    "Date range:",
    df[TIMESTAMP_COL].min(),
    "to",
    df[TIMESTAMP_COL].max(),
)
print(
    "Overall fraud rate:",
    f"{df[TARGET_COL].mean():.4%}",
)


# ============================================================
# 3. 数据概览与时间划分审计
# ============================================================

overview_rows = [
    {
        "metric": "row_count",
        "value": len(df),
    },
    {
        "metric": "column_count",
        "value": df.shape[1],
    },
    {
        "metric": "start_timestamp",
        "value": df[TIMESTAMP_COL].min(),
    },
    {
        "metric": "end_timestamp",
        "value": df[TIMESTAMP_COL].max(),
    },
    {
        "metric": "fraud_count",
        "value": int(df[TARGET_COL].sum()),
    },
    {
        "metric": "fraud_rate",
        "value": float(df[TARGET_COL].mean()),
    },
    {
        "metric": "unique_users",
        "value": int(df[USER_COL].nunique()),
    },
    {
        "metric": "duplicate_transaction_ids",
        "value": int(
            df["transaction_id"].duplicated().sum()
        ),
    },
]

data_overview = pd.DataFrame(overview_rows)
data_overview.to_csv(
    OUTPUT_DIR / "feature_leakage_data_overview.csv",
    index=False,
)

temporal_split_summary = (
    df.groupby("temporal_split")
    .agg(
        rows=(TARGET_COL, "size"),
        fraud_count=(TARGET_COL, "sum"),
        fraud_rate=(TARGET_COL, "mean"),
        unique_users=(USER_COL, "nunique"),
        start_timestamp=(TIMESTAMP_COL, "min"),
        end_timestamp=(TIMESTAMP_COL, "max"),
    )
    .reset_index()
)

split_order = {
    "train": 0,
    "validation": 1,
    "test": 2,
}
temporal_split_summary["split_order"] = (
    temporal_split_summary["temporal_split"]
    .map(split_order)
)
temporal_split_summary = (
    temporal_split_summary
    .sort_values("split_order")
    .drop(columns="split_order")
)

temporal_split_summary.to_csv(
    OUTPUT_DIR / "temporal_split_summary.csv",
    index=False,
)


# ============================================================
# 4. 用户跨集合重叠检查
# ============================================================

users_by_split = {
    split_name: set(
        df.loc[
            df["temporal_split"] == split_name,
            USER_COL,
        ].astype(str)
    )
    for split_name in [
        "train",
        "validation",
        "test",
    ]
}

user_overlap_summary = pd.DataFrame(
    [
        {
            "comparison": "train_vs_validation",
            "left_unique_users": len(users_by_split["train"]),
            "right_unique_users": len(users_by_split["validation"]),
            "overlap_users": len(
                users_by_split["train"]
                & users_by_split["validation"]
            ),
        },
        {
            "comparison": "train_vs_test",
            "left_unique_users": len(users_by_split["train"]),
            "right_unique_users": len(users_by_split["test"]),
            "overlap_users": len(
                users_by_split["train"]
                & users_by_split["test"]
            ),
        },
        {
            "comparison": "validation_vs_test",
            "left_unique_users": len(users_by_split["validation"]),
            "right_unique_users": len(users_by_split["test"]),
            "overlap_users": len(
                users_by_split["validation"]
                & users_by_split["test"]
            ),
        },
    ]
)

user_overlap_summary["overlap_rate_left"] = (
    user_overlap_summary["overlap_users"]
    / user_overlap_summary["left_unique_users"]
)
user_overlap_summary["overlap_rate_right"] = (
    user_overlap_summary["overlap_users"]
    / user_overlap_summary["right_unique_users"]
)

user_overlap_summary.to_csv(
    OUTPUT_DIR / "temporal_user_overlap_summary.csv",
    index=False,
)


# ============================================================
# 5. failed_transaction_count_7d 规则审计
# ============================================================

failed_count_summary = build_group_fraud_summary(
    df,
    FAILED_COUNT_COL,
)

failed_count_summary.to_csv(
    OUTPUT_DIR / "failed_count_fraud_rate_audit.csv",
    index=False,
)

rule_prediction = (
    df[FAILED_COUNT_COL] >= RULE_THRESHOLD
).astype(int)

rule_metrics_rows: list[dict[str, float | int | str]] = []

for split_name in [
    "all",
    "train",
    "validation",
    "test",
]:
    if split_name == "all":
        subset = df
        subset_prediction = rule_prediction
    else:
        mask = df["temporal_split"] == split_name
        subset = df.loc[mask]
        subset_prediction = rule_prediction.loc[mask]

    metrics = classification_metrics(
        subset[TARGET_COL],
        subset_prediction,
        subset_prediction.astype(float),
    )

    metrics["split"] = split_name
    metrics["rule"] = (
        f"{FAILED_COUNT_COL} >= {RULE_THRESHOLD}"
    )
    rule_metrics_rows.append(metrics)

rule_metrics = pd.DataFrame(rule_metrics_rows)

column_order = [
    "split",
    "rule",
    "rows",
    "fraud_count",
    "fraud_rate",
    "predicted_positive_count",
    "review_rate",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
]

rule_metrics = rule_metrics[column_order]

rule_metrics.to_csv(
    OUTPUT_DIR / "failed_count_rule_baseline_metrics.csv",
    index=False,
)


# ============================================================
# 6. 规则在不同月份与不同集合中的稳定性
# ============================================================

df["year_month"] = (
    df[TIMESTAMP_COL]
    .dt.to_period("M")
    .astype(str)
)

monthly_rule_rows = []

for month_value, month_df in df.groupby("year_month"):
    month_prediction = (
        month_df[FAILED_COUNT_COL] >= RULE_THRESHOLD
    ).astype(int)

    month_metrics = classification_metrics(
        month_df[TARGET_COL],
        month_prediction,
        month_prediction.astype(float),
    )

    month_metrics["year_month"] = month_value
    monthly_rule_rows.append(month_metrics)

monthly_rule_metrics = pd.DataFrame(monthly_rule_rows)

monthly_rule_metrics.to_csv(
    OUTPUT_DIR / "monthly_rule_stability.csv",
    index=False,
)


# ============================================================
# 7. 特征代理风险审计
# ============================================================

excluded_from_feature_audit = {
    TARGET_COL,
    "transaction_id",
    USER_COL,
    TIMESTAMP_COL,
    "temporal_split",
    "year_month",
}

feature_audit_rows = []

for feature in df.columns:
    if feature in excluded_from_feature_audit:
        continue

    if pd.api.types.is_numeric_dtype(df[feature]):
        feature_audit_rows.append(
            audit_numeric_feature(df, feature)
        )
    else:
        feature_audit_rows.append(
            audit_categorical_feature(df, feature)
        )

feature_audit = pd.DataFrame(feature_audit_rows)

# 数值字段按与目标的绝对相关性排序；
# 类别字段按组间欺诈率差异排序。
feature_audit["numeric_risk_score"] = (
    feature_audit["target_correlation"]
    .abs()
    .fillna(-1)
)

feature_audit["categorical_risk_score"] = (
    feature_audit["fraud_rate_range"]
    .fillna(-1)
)

feature_audit["proxy_risk_score"] = feature_audit[
    [
        "numeric_risk_score",
        "categorical_risk_score",
    ]
].max(axis=1)

feature_audit = (
    feature_audit
    .sort_values(
        "proxy_risk_score",
        ascending=False,
    )
    .drop(
        columns=[
            "numeric_risk_score",
            "categorical_risk_score",
        ]
    )
)

feature_audit.to_csv(
    OUTPUT_DIR / "potential_proxy_feature_audit.csv",
    index=False,
)


# ============================================================
# 8. 关键二元字段组合审计
# ============================================================

binary_combination_summary = (
    df.groupby(
        [
            PREVIOUS_FRAUD_COL,
            IP_FLAG_COL,
            FAILED_COUNT_COL,
        ],
        dropna=False,
    )[TARGET_COL]
    .agg(
        transaction_count="size",
        fraud_count="sum",
        fraud_rate="mean",
    )
    .reset_index()
    .sort_values(
        [
            "fraud_rate",
            "transaction_count",
        ],
        ascending=[False, False],
    )
)

binary_combination_summary.to_csv(
    OUTPUT_DIR / "key_risk_factor_combination_audit.csv",
    index=False,
)


# ============================================================
# 9. 绘图
# ============================================================

plt.figure(figsize=(10, 6))

plot_df = failed_count_summary.sort_values(
    FAILED_COUNT_COL
)

bars = plt.bar(
    plot_df[FAILED_COUNT_COL].astype(str),
    plot_df["fraud_rate_pct"],
)

plt.axhline(
    df[TARGET_COL].mean() * 100,
    linestyle="--",
    linewidth=1.5,
    label="Overall fraud rate",
)

plt.title(
    "Fraud Rate by Failed Transaction Count (7 Days)"
)
plt.xlabel(
    "Failed transaction count in previous 7 days"
)
plt.ylabel("Fraud rate (%)")
plt.legend()

for bar, rate in zip(
    bars,
    plot_df["fraud_rate_pct"],
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{rate:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.savefig(
    FIGURE_DIR
    / "35_failed_count_fraud_rate_audit.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()


plt.figure(figsize=(10, 6))

monthly_plot = monthly_rule_metrics.sort_values(
    "year_month"
)

plt.plot(
    monthly_plot["year_month"],
    monthly_plot["precision"],
    marker="o",
    label="Precision",
)

plt.plot(
    monthly_plot["year_month"],
    monthly_plot["recall"],
    marker="o",
    label="Recall",
)

plt.plot(
    monthly_plot["year_month"],
    monthly_plot["review_rate"],
    marker="o",
    label="Review rate",
)

plt.title(
    "Monthly Stability of the Failed-Count Rule"
)
plt.xlabel("Month")
plt.ylabel("Metric")
plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()
plt.savefig(
    FIGURE_DIR
    / "36_failed_count_rule_stability.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 10. 终端摘要
# ============================================================

print()
print("Temporal split summary:")
print(
    temporal_split_summary.to_string(
        index=False,
    )
)

print()
print("User overlap summary:")
print(
    user_overlap_summary.to_string(
        index=False,
    )
)

print()
print("Fraud rate by failed transaction count:")
print(
    failed_count_summary.to_string(
        index=False,
    )
)

print()
print("Rule baseline metrics:")
print(
    rule_metrics[
        [
            "split",
            "precision",
            "recall",
            "f1",
            "review_rate",
            "false_positive",
            "false_negative",
        ]
    ].to_string(
        index=False,
    )
)

print()
print("Top potential proxy features:")
print(
    feature_audit.head(10).to_string(
        index=False,
    )
)

print()
print("Saved audit outputs to:")
print(OUTPUT_DIR)
print()
print("Feature leakage audit completed successfully.")
