from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 0. 路径与可修改参数
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "fraud_data_cleaned.csv"
)

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIAGNOSTIC_MODEL_FILE = (
    MODELS_DIR
    / "residual_logistic_diagnostic_pipeline.pkl"
)

FEATURE_FILE = (
    MODELS_DIR
    / "diagnostic_model_feature_columns.json"
)

POLICY_FILE = (
    MODELS_DIR
    / "decision_policy.json"
)

METADATA_FILE = (
    MODELS_DIR
    / "model_metadata.json"
)

SAMPLE_INPUT_FILE = (
    OUTPUT_DIR
    / "sample_transaction_input.json"
)

ARTIFACT_SUMMARY_FILE = (
    OUTPUT_DIR
    / "diagnostic_model_artifact_summary.csv"
)

TARGET_COL = "fraud_label"
TIMESTAMP_COL = "timestamp"

FAILED_COUNT_COL = "failed_transaction_count_7d"
RULE_THRESHOLD = 4

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
RANDOM_STATE = 42

# 这些字段永远不进入诊断模型。
# failed_transaction_count_7d 已被确认是合成数据中的确定性代理，
# 所以只用于透明规则层，不用于残余机器学习模型。
EXCLUDED_COLUMNS = {
    "transaction_id",
    "user_id",
    "timestamp",
    "risk_score",
    "failed_transaction_count_7d",
    "fraud_label",
    "year",
}


# ============================================================
# 1. 工具函数
# ============================================================

def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def temporal_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    ordered = (
        df.sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
        .copy()
    )

    row_count = len(ordered)
    train_end = int(row_count * TRAIN_RATIO)
    validation_end = int(
        row_count
        * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_df = ordered.iloc[:train_end].copy()
    validation_df = (
        ordered.iloc[
            train_end:validation_end
        ].copy()
    )
    test_df = (
        ordered.iloc[
            validation_end:
        ].copy()
    )

    return train_df, validation_df, test_df


def safe_roc_auc(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    if y_true.nunique() < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )


def ranking_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    fraud_rate = float(y_true.mean())
    pr_auc = float(
        average_precision_score(
            y_true,
            probabilities,
        )
    )

    absolute_uplift = pr_auc - fraud_rate

    relative_uplift = (
        absolute_uplift / fraud_rate
        if fraud_rate > 0
        else float("nan")
    )

    return {
        "rows": int(len(y_true)),
        "fraud_count": int(y_true.sum()),
        "fraud_rate": fraud_rate,
        "roc_auc": safe_roc_auc(
            y_true,
            probabilities,
        ),
        "pr_auc": pr_auc,
        "absolute_pr_auc_uplift": (
            absolute_uplift
        ),
        "relative_pr_auc_uplift": (
            relative_uplift
        ),
    }


def to_python_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


# ============================================================
# 2. 读取数据并检查
# ============================================================

print("=" * 74)
print("Governance-Safe Diagnostic Model Artifact")
print("=" * 74)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

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
    invalid_count = int(
        df[TIMESTAMP_COL]
        .isna()
        .sum()
    )

    raise ValueError(
        f"{invalid_count} invalid timestamps found."
    )

if not set(
    df[TARGET_COL]
    .dropna()
    .unique()
).issubset({0, 1}):
    raise ValueError(
        f"{TARGET_COL} must contain only 0 and 1."
    )


# ============================================================
# 3. 时间顺序划分与残余交易
# ============================================================

train_full, validation_full, test_full = (
    temporal_split(df)
)

train_residual = train_full.loc[
    train_full[FAILED_COUNT_COL]
    < RULE_THRESHOLD
].copy()

validation_residual = (
    validation_full.loc[
        validation_full[FAILED_COUNT_COL]
        < RULE_THRESHOLD
    ].copy()
)

test_residual = test_full.loc[
    test_full[FAILED_COUNT_COL]
    < RULE_THRESHOLD
].copy()

print()
print("Residual split sizes:")
print(
    f"Train: {len(train_residual):,}"
)
print(
    f"Validation: {len(validation_residual):,}"
)
print(
    f"Test: {len(test_residual):,}"
)


# ============================================================
# 4. 诊断模型特征
# ============================================================

feature_columns = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]

X_train = train_residual[
    feature_columns
]
y_train = train_residual[
    TARGET_COL
]

X_validation = validation_residual[
    feature_columns
]
y_validation = validation_residual[
    TARGET_COL
]

X_test = test_residual[
    feature_columns
]
y_test = test_residual[
    TARGET_COL
]

numeric_columns = (
    X_train
    .select_dtypes(
        include=["number", "bool"]
    )
    .columns
    .tolist()
)

categorical_columns = [
    column
    for column in feature_columns
    if column not in numeric_columns
]

print()
print(
    "Diagnostic feature count:",
    len(feature_columns),
)
print(
    "Numeric features:",
    len(numeric_columns),
)
print(
    "Categorical features:",
    len(categorical_columns),
)


# ============================================================
# 5. 预处理与 Logistic Regression
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
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
            SimpleImputer(
                strategy="most_frequent",
            ),
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

diagnostic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# ============================================================
# 6. 训练并复现 Validation / Test 排序表现
# ============================================================

print()
print(
    "Training residual diagnostic model "
    "on chronological Train only..."
)

diagnostic_model.fit(
    X_train,
    y_train,
)

validation_probabilities = (
    diagnostic_model.predict_proba(
        X_validation
    )[:, 1]
)

test_probabilities = (
    diagnostic_model.predict_proba(
        X_test
    )[:, 1]
)

validation_metrics = ranking_metrics(
    y_validation,
    validation_probabilities,
)

test_metrics = ranking_metrics(
    y_test,
    test_probabilities,
)

print()
print("Validation ranking metrics:")
for key, value in validation_metrics.items():
    if isinstance(value, float):
        print(
            f"{key}: {value:.6f}"
        )
    else:
        print(
            f"{key}: {value}"
        )

print()
print("Independent Test ranking metrics:")
for key, value in test_metrics.items():
    if isinstance(value, float):
        print(
            f"{key}: {value:.6f}"
        )
    else:
        print(
            f"{key}: {value}"
        )


# ============================================================
# 7. 保存诊断模型
# ============================================================

# 注意：
# 该模型只用于演示 API / Dashboard 工程与模型风险治理。
# 它不是生产部署模型，也不能触发自动批准、拒绝或拦截。
joblib.dump(
    diagnostic_model,
    DIAGNOSTIC_MODEL_FILE,
)

print()
print(
    "Diagnostic model saved to:",
    DIAGNOSTIC_MODEL_FILE,
)


# ============================================================
# 8. 保存特征清单
# ============================================================

feature_info = {
    "target_column": TARGET_COL,
    "feature_columns": feature_columns,
    "numeric_columns": numeric_columns,
    "categorical_columns": (
        categorical_columns
    ),
    "excluded_columns": sorted(
        EXCLUDED_COLUMNS
    ),
    "failed_count_rule_column": (
        FAILED_COUNT_COL
    ),
    "failed_count_rule_threshold": (
        RULE_THRESHOLD
    ),
    "artifact_role": (
        "diagnostic_only"
    ),
}

FEATURE_FILE.write_text(
    json.dumps(
        feature_info,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# 9. 保存透明决策策略
# ============================================================

decision_policy = {
    "policy_name": (
        "Synthetic Rule and Diagnostic "
        "Model Governance Policy"
    ),
    "policy_version": "2.0",
    "primary_decision_layer": {
        "type": "transparent_rule",
        "feature": FAILED_COUNT_COL,
        "operator": ">=",
        "threshold": RULE_THRESHOLD,
        "trigger_label": (
            "Synthetic Rule Alert"
        ),
        "recommended_action": (
            "Manual review for portfolio "
            "demonstration only"
        ),
        "production_eligible": False,
    },
    "secondary_layer": {
        "type": (
            "residual_logistic_diagnostic_score"
        ),
        "model_file": (
            "models/"
            "residual_logistic_"
            "diagnostic_pipeline.pkl"
        ),
        "decision_authority": False,
        "production_eligible": False,
        "allowed_use": [
            "engineering demonstration",
            "model-risk diagnostics",
            "API and dashboard integration",
        ],
        "prohibited_use": [
            "automatic approval",
            "automatic rejection",
            "transaction blocking",
            "customer-impact decisions",
        ],
    },
    "governance_warning": (
        "The dataset is synthetic. "
        "The deterministic failed-count rule "
        "must not be interpreted as evidence "
        "of real banking fraud performance."
    ),
}

POLICY_FILE.write_text(
    json.dumps(
        decision_policy,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# 10. 保存治理元数据
# ============================================================

metadata = {
    "project_name": (
        "Financial Transaction Risk Monitoring"
    ),
    "artifact_name": (
        "Residual Logistic Diagnostic Pipeline"
    ),
    "artifact_role": "diagnostic_only",
    "deployment_eligible": False,
    "automatic_decision_approved": False,
    "model_file": (
        "models/"
        "residual_logistic_"
        "diagnostic_pipeline.pkl"
    ),
    "feature_file": (
        "models/"
        "diagnostic_model_"
        "feature_columns.json"
    ),
    "decision_policy_file": (
        "models/decision_policy.json"
    ),
    "training_data": (
        "data/processed/"
        "fraud_data_cleaned.csv"
    ),
    "training_scope": (
        "chronological_train_split_"
        "residual_transactions_only"
    ),
    "number_of_training_rows": int(
        len(train_residual)
    ),
    "number_of_features": int(
        len(feature_columns)
    ),
    "target_column": TARGET_COL,
    "risk_score_excluded": True,
    "failed_count_excluded_from_ml": True,
    "failed_count_rule_threshold": (
        RULE_THRESHOLD
    ),
    "validation_metrics": (
        validation_metrics
    ),
    "independent_test_metrics": (
        test_metrics
    ),
    "deployment_decision": (
        "do_not_deploy_residual_model"
    ),
    "deployment_reason": (
        "The residual diagnostic model "
        "does not provide sufficient "
        "out-of-time predictive signal."
    ),
    "governance_warning": (
        "Synthetic portfolio dataset only. "
        "No automatic customer-impact action "
        "is approved."
    ),
}

METADATA_FILE.write_text(
    json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# 11. 保存 API 示例输入
# ============================================================

sample_source = (
    test_full.iloc[0]
    if len(test_full) > 0
    else df.iloc[0]
)

sample_input = {
    "transaction_amount": to_python_value(
        sample_source[
            "transaction_amount"
        ]
    ),
    "transaction_type": to_python_value(
        sample_source[
            "transaction_type"
        ]
    ),
    "timestamp": pd.Timestamp(
        sample_source[TIMESTAMP_COL]
    ).isoformat(),
    "account_balance": to_python_value(
        sample_source[
            "account_balance"
        ]
    ),
    "device_type": to_python_value(
        sample_source[
            "device_type"
        ]
    ),
    "location": to_python_value(
        sample_source[
            "location"
        ]
    ),
    "merchant_category": to_python_value(
        sample_source[
            "merchant_category"
        ]
    ),
    "ip_address_flag": to_python_value(
        sample_source[
            "ip_address_flag"
        ]
    ),
    "previous_fraudulent_activity": (
        to_python_value(
            sample_source[
                "previous_fraudulent_activity"
            ]
        )
    ),
    "daily_transaction_count": (
        to_python_value(
            sample_source[
                "daily_transaction_count"
            ]
        )
    ),
    "avg_transaction_amount_7d": (
        to_python_value(
            sample_source[
                "avg_transaction_amount_7d"
            ]
        )
    ),
    "failed_transaction_count_7d": (
        to_python_value(
            sample_source[
                FAILED_COUNT_COL
            ]
        )
    ),
    "card_type": to_python_value(
        sample_source[
            "card_type"
        ]
    ),
    "card_age": to_python_value(
        sample_source[
            "card_age"
        ]
    ),
    "transaction_distance": (
        to_python_value(
            sample_source[
                "transaction_distance"
            ]
        )
    ),
    "authentication_method": (
        to_python_value(
            sample_source[
                "authentication_method"
            ]
        )
    ),
}

SAMPLE_INPUT_FILE.write_text(
    json.dumps(
        sample_input,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# ============================================================
# 12. 保存摘要
# ============================================================

artifact_summary = pd.DataFrame(
    [
        {
            "artifact": (
                "Residual Logistic "
                "Diagnostic Pipeline"
            ),
            "artifact_role": (
                "diagnostic_only"
            ),
            "deployment_eligible": False,
            "training_rows": int(
                len(train_residual)
            ),
            "feature_count": int(
                len(feature_columns)
            ),
            "validation_roc_auc": (
                validation_metrics[
                    "roc_auc"
                ]
            ),
            "validation_pr_auc": (
                validation_metrics[
                    "pr_auc"
                ]
            ),
            "test_roc_auc": (
                test_metrics[
                    "roc_auc"
                ]
            ),
            "test_pr_auc": (
                test_metrics[
                    "pr_auc"
                ]
            ),
            "test_fraud_rate": (
                test_metrics[
                    "fraud_rate"
                ]
            ),
            "deployment_decision": (
                "do_not_deploy_"
                "residual_model"
            ),
        }
    ]
)

artifact_summary.to_csv(
    ARTIFACT_SUMMARY_FILE,
    index=False,
)

print()
print("Saved files:")
print(DIAGNOSTIC_MODEL_FILE)
print(FEATURE_FILE)
print(POLICY_FILE)
print(METADATA_FILE)
print(SAMPLE_INPUT_FILE)
print(ARTIFACT_SUMMARY_FILE)

print()
print(
    "Diagnostic artifact creation "
    "completed successfully."
)
