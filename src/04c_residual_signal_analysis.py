from __future__ import annotations

from pathlib import Path
import json
import warnings

import matplotlib

# 终端运行时不弹出绘图窗口
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
FAILED_COUNT_COL = "failed_transaction_count_7d"

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
RANDOM_STATE = 42

# 第一层确定性规则
RULE_THRESHOLD = 4

# 第二层残余模型的业务容量情景：
# 分别在 Validation 残余交易中选取最高风险的 5%、10%、20%，
# 再把对应阈值原样应用到 Test。
RESIDUAL_REVIEW_SHARES = [0.05, 0.10, 0.20]

# 判断残余模型是否具备部署价值的最低标准
MIN_TEST_ROC_AUC = 0.55
MIN_RELATIVE_PR_AUC_UPLIFT = 0.10

# 永远不进入残余模型的字段
NON_FEATURE_COLUMNS = {
    "transaction_id",
    "user_id",
    "timestamp",
    "fraud_label",
    "risk_score",  # 来源不透明，排除
    "failed_transaction_count_7d",  # 第一层规则字段，残余层排除
    "year",  # 当前数据只有 2023 年，为常量
}


# ============================================================
# 1. 工具函数
# ============================================================

def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns: " + ", ".join(missing)
        )


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
                OneHotEncoder(handle_unknown="ignore"),
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


def threshold_for_top_share(
    probabilities: np.ndarray,
    review_share: float,
) -> float:
    if not 0 < review_share < 1:
        raise ValueError(
            "review_share must be between 0 and 1."
        )

    return float(
        np.quantile(
            probabilities,
            1 - review_share,
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
            accuracy_score(y_true, predictions)
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


def audit_numeric_feature(
    df: pd.DataFrame,
    feature: str,
) -> dict[str, float | int | str]:
    valid = df[[feature, TARGET_COL]].dropna()

    correlation = valid[feature].corr(
        valid[TARGET_COL]
    )

    if valid[feature].nunique() > 1:
        raw_auc = safe_roc_auc(
            valid[TARGET_COL],
            valid[feature].to_numpy(),
        )
        oriented_auc = max(
            raw_auc,
            1 - raw_auc,
        )
    else:
        raw_auc = np.nan
        oriented_auc = np.nan

    fraud_mean = valid.loc[
        valid[TARGET_COL] == 1,
        feature,
    ].mean()

    non_fraud_mean = valid.loc[
        valid[TARGET_COL] == 0,
        feature,
    ].mean()

    return {
        "feature": feature,
        "feature_type": "numeric",
        "unique_values": int(
            valid[feature].nunique()
        ),
        "non_fraud_mean": float(non_fraud_mean),
        "fraud_mean": float(fraud_mean),
        "mean_difference": float(
            fraud_mean - non_fraud_mean
        ),
        "target_correlation": (
            float(correlation)
            if pd.notna(correlation)
            else np.nan
        ),
        "raw_univariate_roc_auc": raw_auc,
        "oriented_univariate_roc_auc": oriented_auc,
        "minimum_group_fraud_rate": np.nan,
        "maximum_group_fraud_rate": np.nan,
        "fraud_rate_range": np.nan,
    }


def audit_categorical_feature(
    df: pd.DataFrame,
    feature: str,
) -> dict[str, float | int | str]:
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
        "unique_values": int(
            df[feature].nunique(dropna=False)
        ),
        "non_fraud_mean": np.nan,
        "fraud_mean": np.nan,
        "mean_difference": np.nan,
        "target_correlation": np.nan,
        "raw_univariate_roc_auc": np.nan,
        "oriented_univariate_roc_auc": np.nan,
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


def combined_strategy_metrics(
    full_test_df: pd.DataFrame,
    residual_test_index: pd.Index,
    residual_probabilities: np.ndarray,
    residual_threshold: float | None,
    scenario_name: str,
) -> dict[str, float | int | str]:
    rule_flags = (
        full_test_df[FAILED_COUNT_COL]
        >= RULE_THRESHOLD
    ).astype(int)

    combined_flags = rule_flags.copy()

    if residual_threshold is not None:
        residual_flags = (
            residual_probabilities
            >= residual_threshold
        ).astype(int)

        combined_flags.loc[
            residual_test_index
        ] = residual_flags

    y_true = full_test_df[TARGET_COL]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        combined_flags,
        labels=[0, 1],
    ).ravel()

    residual_mask = (
        full_test_df[FAILED_COUNT_COL]
        < RULE_THRESHOLD
    )

    residual_reviewed = int(
        combined_flags.loc[residual_mask].sum()
    )

    residual_fraud_captured = int(
        (
            combined_flags.loc[residual_mask]
            * y_true.loc[residual_mask]
        ).sum()
    )

    residual_incremental_precision = (
        residual_fraud_captured
        / residual_reviewed
        if residual_reviewed > 0
        else np.nan
    )

    return {
        "scenario": scenario_name,
        "rows": int(len(full_test_df)),
        "fraud_count": int(y_true.sum()),
        "review_count": int(combined_flags.sum()),
        "review_rate": float(combined_flags.mean()),
        "precision": float(
            precision_score(
                y_true,
                combined_flags,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                combined_flags,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                combined_flags,
                zero_division=0,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "residual_reviewed": residual_reviewed,
        "residual_fraud_captured": residual_fraud_captured,
        "residual_incremental_precision": (
            float(residual_incremental_precision)
            if pd.notna(residual_incremental_precision)
            else np.nan
        ),
    }


# ============================================================
# 2. 读取数据与全量时间划分
# ============================================================

print("=" * 76)
print("Residual Fraud Signal Analysis")
print("=" * 76)

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
        FAILED_COUNT_COL,
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

train_full, validation_full, test_full = (
    temporal_split(df)
)

# 第一层规则覆盖之外的残余交易
train_residual = train_full.loc[
    train_full[FAILED_COUNT_COL] < RULE_THRESHOLD
].copy()

validation_residual = validation_full.loc[
    validation_full[FAILED_COUNT_COL] < RULE_THRESHOLD
].copy()

test_residual = test_full.loc[
    test_full[FAILED_COUNT_COL] < RULE_THRESHOLD
].copy()


# ============================================================
# 3. 残余数据概览
# ============================================================

residual_split_summary = pd.DataFrame(
    [
        {
            "split": "train",
            "rows": len(train_residual),
            "fraud_count": int(
                train_residual[TARGET_COL].sum()
            ),
            "fraud_rate": float(
                train_residual[TARGET_COL].mean()
            ),
            "start_timestamp": train_residual[
                TIMESTAMP_COL
            ].min(),
            "end_timestamp": train_residual[
                TIMESTAMP_COL
            ].max(),
        },
        {
            "split": "validation",
            "rows": len(validation_residual),
            "fraud_count": int(
                validation_residual[TARGET_COL].sum()
            ),
            "fraud_rate": float(
                validation_residual[TARGET_COL].mean()
            ),
            "start_timestamp": validation_residual[
                TIMESTAMP_COL
            ].min(),
            "end_timestamp": validation_residual[
                TIMESTAMP_COL
            ].max(),
        },
        {
            "split": "test",
            "rows": len(test_residual),
            "fraud_count": int(
                test_residual[TARGET_COL].sum()
            ),
            "fraud_rate": float(
                test_residual[TARGET_COL].mean()
            ),
            "start_timestamp": test_residual[
                TIMESTAMP_COL
            ].min(),
            "end_timestamp": test_residual[
                TIMESTAMP_COL
            ].max(),
        },
    ]
)

residual_split_summary.to_csv(
    OUTPUT_DIR / "residual_split_summary.csv",
    index=False,
)

print()
print("Residual split summary:")
print(
    residual_split_summary.to_string(
        index=False,
    )
)


# ============================================================
# 4. 只在 Train 残余数据中做单变量特征审计
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]

feature_audit_rows = []

for feature in feature_columns:
    if pd.api.types.is_numeric_dtype(
        train_residual[feature]
    ):
        feature_audit_rows.append(
            audit_numeric_feature(
                train_residual,
                feature,
            )
        )
    else:
        feature_audit_rows.append(
            audit_categorical_feature(
                train_residual,
                feature,
            )
        )

residual_feature_audit = pd.DataFrame(
    feature_audit_rows
)

residual_feature_audit["numeric_signal_score"] = (
    residual_feature_audit[
        "oriented_univariate_roc_auc"
    ]
    .sub(0.5)
    .abs()
    .fillna(-1)
)

residual_feature_audit["categorical_signal_score"] = (
    residual_feature_audit[
        "fraud_rate_range"
    ]
    .fillna(-1)
)

residual_feature_audit["signal_score"] = (
    residual_feature_audit[
        [
            "numeric_signal_score",
            "categorical_signal_score",
        ]
    ].max(axis=1)
)

residual_feature_audit = (
    residual_feature_audit
    .sort_values(
        "signal_score",
        ascending=False,
    )
    .drop(
        columns=[
            "numeric_signal_score",
            "categorical_signal_score",
        ]
    )
)

residual_feature_audit.to_csv(
    OUTPUT_DIR / "residual_feature_signal_audit.csv",
    index=False,
)

print()
print("Top residual single-feature signals:")
print(
    residual_feature_audit.head(12).to_string(
        index=False,
    )
)


# ============================================================
# 5. 在 Validation 上比较残余模型
# ============================================================

X_train = train_residual[feature_columns]
y_train = train_residual[TARGET_COL]

X_validation = validation_residual[
    feature_columns
]
y_validation = validation_residual[TARGET_COL]

positive_count = int(y_train.sum())
negative_count = int(
    len(y_train) - positive_count
)

models = build_models(
    positive_count=positive_count,
    negative_count=negative_count,
)

validation_rows: list[
    dict[str, float | int | str]
] = []

trained_models: dict[str, Pipeline] = {}

for model_name, estimator in models.items():
    print()
    print(f"Training residual model: {model_name}")

    preprocessor, numeric_columns, categorical_columns = (
        build_preprocessor(X_train)
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

    validation_probabilities = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    baseline_pr_auc = float(
        y_validation.mean()
    )
    model_pr_auc = float(
        average_precision_score(
            y_validation,
            validation_probabilities,
        )
    )
    model_roc_auc = safe_roc_auc(
        y_validation,
        validation_probabilities,
    )

    validation_rows.append(
        {
            "model": model_name,
            "rows": len(y_validation),
            "fraud_rate": baseline_pr_auc,
            "roc_auc": model_roc_auc,
            "pr_auc": model_pr_auc,
            "absolute_pr_auc_uplift": (
                model_pr_auc - baseline_pr_auc
            ),
            "relative_pr_auc_uplift": (
                (
                    model_pr_auc - baseline_pr_auc
                )
                / baseline_pr_auc
                if baseline_pr_auc > 0
                else np.nan
            ),
            "numeric_feature_count": len(
                numeric_columns
            ),
            "categorical_feature_count": len(
                categorical_columns
            ),
        }
    )

    trained_models[model_name] = pipeline

validation_model_comparison = pd.DataFrame(
    validation_rows
).sort_values(
    [
        "pr_auc",
        "roc_auc",
    ],
    ascending=False,
)

validation_model_comparison.to_csv(
    OUTPUT_DIR
    / "residual_validation_model_comparison.csv",
    index=False,
)

print()
print("Residual Validation model comparison:")
print(
    validation_model_comparison.to_string(
        index=False,
    )
)


# ============================================================
# 6. 只用 Validation 选择诊断模型
# ============================================================

selected_row = validation_model_comparison.iloc[0]
selected_model_name = str(
    selected_row["model"]
)
selected_pipeline = trained_models[
    selected_model_name
]

validation_probabilities = (
    selected_pipeline.predict_proba(
        X_validation
    )[:, 1]
)

validation_thresholds = {
    review_share: threshold_for_top_share(
        validation_probabilities,
        review_share,
    )
    for review_share in RESIDUAL_REVIEW_SHARES
}

selected_model_summary = pd.DataFrame(
    [
        {
            "selected_model": selected_model_name,
            "selection_split": "validation",
            "selection_metric": "pr_auc",
            "validation_fraud_rate": float(
                selected_row["fraud_rate"]
            ),
            "validation_roc_auc": float(
                selected_row["roc_auc"]
            ),
            "validation_pr_auc": float(
                selected_row["pr_auc"]
            ),
            "validation_relative_pr_auc_uplift": float(
                selected_row[
                    "relative_pr_auc_uplift"
                ]
            ),
            "deployment_status": (
                "diagnostic_candidate_only"
            ),
        }
    ]
)

selected_model_summary.to_csv(
    OUTPUT_DIR / "residual_selected_model.csv",
    index=False,
)


# ============================================================
# 7. Independent Test 只评估一次
# ============================================================

X_test = test_residual[feature_columns]
y_test = test_residual[TARGET_COL]

test_probabilities = (
    selected_pipeline.predict_proba(
        X_test
    )[:, 1]
)

test_fraud_rate = float(y_test.mean())
test_pr_auc = float(
    average_precision_score(
        y_test,
        test_probabilities,
    )
)
test_roc_auc = safe_roc_auc(
    y_test,
    test_probabilities,
)

absolute_pr_auc_uplift = (
    test_pr_auc - test_fraud_rate
)

relative_pr_auc_uplift = (
    absolute_pr_auc_uplift / test_fraud_rate
    if test_fraud_rate > 0
    else np.nan
)

residual_test_metrics = pd.DataFrame(
    [
        {
            "model": selected_model_name,
            "rows": len(y_test),
            "fraud_count": int(y_test.sum()),
            "fraud_rate": test_fraud_rate,
            "roc_auc": test_roc_auc,
            "pr_auc": test_pr_auc,
            "absolute_pr_auc_uplift": absolute_pr_auc_uplift,
            "relative_pr_auc_uplift": (
                relative_pr_auc_uplift
            ),
            "selection_split": "validation",
            "test_used_for_selection": False,
        }
    ]
)

residual_test_metrics.to_csv(
    OUTPUT_DIR / "residual_independent_test_metrics.csv",
    index=False,
)

print()
print("Residual Independent Test ranking metrics:")
print(
    residual_test_metrics.to_string(
        index=False,
    )
)


# ============================================================
# 8. 残余模型 Top-K 审核能力
# ============================================================

residual_lift_rows = []

for review_share, threshold in (
    validation_thresholds.items()
):
    metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    metrics["model"] = selected_model_name
    metrics["target_residual_review_share"] = (
        review_share
    )
    metrics["threshold_source"] = "validation"

    precision_lift = (
        metrics["precision"] / test_fraud_rate
        if test_fraud_rate > 0
        else np.nan
    )

    metrics["precision_lift_vs_random"] = (
        precision_lift
    )

    residual_lift_rows.append(metrics)

residual_test_lift_table = pd.DataFrame(
    residual_lift_rows
)

residual_test_lift_table.to_csv(
    OUTPUT_DIR / "residual_test_lift_table.csv",
    index=False,
)

print()
print("Residual Test capacity scenarios:")
print(
    residual_test_lift_table[
        [
            "target_residual_review_share",
            "threshold",
            "review_rate",
            "precision",
            "recall",
            "f1",
            "precision_lift_vs_random",
        ]
    ].to_string(
        index=False,
    )
)


# ============================================================
# 9. 两层策略：规则层 + 残余模型层
# ============================================================

two_layer_rows = [
    combined_strategy_metrics(
        full_test_df=test_full,
        residual_test_index=test_residual.index,
        residual_probabilities=test_probabilities,
        residual_threshold=None,
        scenario_name="Rule only",
    )
]

for review_share, threshold in (
    validation_thresholds.items()
):
    two_layer_rows.append(
        combined_strategy_metrics(
            full_test_df=test_full,
            residual_test_index=test_residual.index,
            residual_probabilities=test_probabilities,
            residual_threshold=threshold,
            scenario_name=(
                "Rule + residual model "
                f"top {int(review_share * 100)}%"
            ),
        )
    )

two_layer_strategy = pd.DataFrame(
    two_layer_rows
)

two_layer_strategy.to_csv(
    OUTPUT_DIR / "two_layer_test_strategy.csv",
    index=False,
)

print()
print("Two-layer Test strategy:")
print(
    two_layer_strategy[
        [
            "scenario",
            "review_rate",
            "precision",
            "recall",
            "f1",
            "residual_reviewed",
            "residual_fraud_captured",
            "residual_incremental_precision",
        ]
    ].to_string(
        index=False,
    )
)


# ============================================================
# 10. 自动生成结论
# ============================================================

has_deployable_signal = bool(
    test_roc_auc >= MIN_TEST_ROC_AUC
    and relative_pr_auc_uplift
    >= MIN_RELATIVE_PR_AUC_UPLIFT
)

if has_deployable_signal:
    deployment_recommendation = (
        "residual_model_has_limited_but_potential_signal"
    )
    conclusion = (
        "The residual model exceeds the minimum ranking "
        "and PR-AUC uplift criteria. It may be retained as "
        "a secondary review-ranking layer, subject to "
        "calibration and monitoring."
    )
else:
    deployment_recommendation = (
        "do_not_deploy_residual_model"
    )
    conclusion = (
        "After removing the deterministic failed-count "
        "proxy, the remaining features do not provide "
        "sufficient out-of-time predictive signal. Keep "
        "the rule layer as a synthetic-data demonstration "
        "and treat the residual model as diagnostic only."
    )

decision = {
    "selected_residual_model": selected_model_name,
    "test_fraud_rate": test_fraud_rate,
    "test_roc_auc": test_roc_auc,
    "test_pr_auc": test_pr_auc,
    "absolute_pr_auc_uplift": absolute_pr_auc_uplift,
    "relative_pr_auc_uplift": (
        relative_pr_auc_uplift
    ),
    "minimum_required_roc_auc": MIN_TEST_ROC_AUC,
    "minimum_required_relative_pr_auc_uplift": (
        MIN_RELATIVE_PR_AUC_UPLIFT
    ),
    "deployment_recommendation": (
        deployment_recommendation
    ),
    "conclusion": conclusion,
}

with open(
    OUTPUT_DIR / "residual_signal_decision.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        decision,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 11. 绘图
# ============================================================

plot_validation = (
    validation_model_comparison
    .sort_values("pr_auc", ascending=False)
    .copy()
)

plt.figure(figsize=(9, 6))

bars = plt.bar(
    plot_validation["model"],
    plot_validation["pr_auc"],
)

plt.axhline(
    y=float(y_validation.mean()),
    linestyle="--",
    linewidth=1.5,
    label="Residual fraud-rate baseline",
)

plt.title(
    "Residual Validation PR-AUC"
)
plt.xlabel("Model")
plt.ylabel("PR-AUC")
plt.legend()

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
    / "39_residual_validation_pr_auc.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()


plt.figure(figsize=(10, 6))

plt.plot(
    two_layer_strategy["scenario"],
    two_layer_strategy["precision"],
    marker="o",
    label="Precision",
)

plt.plot(
    two_layer_strategy["scenario"],
    two_layer_strategy["recall"],
    marker="o",
    label="Recall",
)

plt.plot(
    two_layer_strategy["scenario"],
    two_layer_strategy["review_rate"],
    marker="o",
    label="Review rate",
)

plt.title(
    "Two-Layer Test Strategy"
)
plt.xlabel("Scenario")
plt.ylabel("Metric")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()
plt.savefig(
    FIGURE_DIR
    / "40_two_layer_test_strategy.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()


# ============================================================
# 12. 终端最终摘要
# ============================================================

print()
print("Decision:")
print(json.dumps(
    decision,
    indent=2,
    ensure_ascii=False,
))

print()
print("Saved outputs to:")
print(OUTPUT_DIR)
print()
print("Residual fraud signal analysis completed successfully.")
