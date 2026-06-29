from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 0. 日志设置
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. 项目路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DIAGNOSTIC_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "residual_logistic_diagnostic_pipeline.pkl"
)

FEATURE_FILE = (
    BASE_DIR
    / "models"
    / "diagnostic_model_feature_columns.json"
)

POLICY_FILE = (
    BASE_DIR
    / "models"
    / "decision_policy.json"
)

METADATA_FILE = (
    BASE_DIR
    / "models"
    / "model_metadata.json"
)


# ============================================================
# 2. 检查并加载治理安全的模型文件
# ============================================================

required_files = [
    DIAGNOSTIC_MODEL_FILE,
    FEATURE_FILE,
    POLICY_FILE,
    METADATA_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise RuntimeError(
            f"Required file not found: {file_path}"
        )

diagnostic_model = joblib.load(
    DIAGNOSTIC_MODEL_FILE
)

feature_info = json.loads(
    FEATURE_FILE.read_text(
        encoding="utf-8"
    )
)

decision_policy = json.loads(
    POLICY_FILE.read_text(
        encoding="utf-8"
    )
)

metadata = json.loads(
    METADATA_FILE.read_text(
        encoding="utf-8"
    )
)

MODEL_FEATURES = feature_info[
    "feature_columns"
]

PRIMARY_RULE = decision_policy[
    "primary_decision_layer"
]

RULE_FEATURE = str(
    PRIMARY_RULE["feature"]
)

RULE_THRESHOLD = int(
    PRIMARY_RULE["threshold"]
)

POLICY_VERSION = str(
    decision_policy.get(
        "policy_version",
        "2.0",
    )
)

ARTIFACT_NAME = str(
    metadata.get(
        "artifact_name",
        "Residual Logistic Diagnostic Pipeline",
    )
)

ARTIFACT_ROLE = str(
    metadata.get(
        "artifact_role",
        "diagnostic_only",
    )
)

DEPLOYMENT_ELIGIBLE = bool(
    metadata.get(
        "deployment_eligible",
        False,
    )
)

AUTOMATIC_DECISION_APPROVED = bool(
    metadata.get(
        "automatic_decision_approved",
        False,
    )
)

GOVERNANCE_WARNING = str(
    decision_policy.get(
        "governance_warning",
        (
            "Synthetic portfolio dataset only. "
            "No automatic customer-impact action is approved."
        ),
    )
)

logger.info(
    "Diagnostic model and governance policy loaded."
)
logger.info(
    "Artifact: %s | role=%s | deployment_eligible=%s",
    ARTIFACT_NAME,
    ARTIFACT_ROLE,
    DEPLOYMENT_ELIGIBLE,
)


# ============================================================
# 3. 创建 FastAPI 应用
# ============================================================

app = FastAPI(
    title=(
        "Financial Transaction Risk "
        "Monitoring Governance API"
    ),
    description=(
        "A governance-aware portfolio API. "
        "It exposes a transparent synthetic-data rule "
        "and an optional residual diagnostic score. "
        "It does not approve, reject, or block transactions."
    ),
    version="2.0.0",
)


# ============================================================
# 4. API 输入格式
# ============================================================

class TransactionRequest(BaseModel):
    """
    输入一笔交易。

    timestamp 会自动拆分为：
    month、day、hour、day_of_week、is_weekend。

    failed_transaction_count_7d 仅用于透明规则层，
    不进入残余诊断模型。
    """

    transaction_amount: float = Field(
        ...,
        ge=0,
        description="Current transaction amount",
    )

    transaction_type: str = Field(
        ...,
        description=(
            "Examples: Online, POS, ATM Withdrawal, "
            "Bank Transfer"
        ),
    )

    timestamp: datetime = Field(
        ...,
        description="Transaction timestamp",
    )

    account_balance: float = Field(
        ...,
        ge=0,
        description="Current account balance",
    )

    device_type: str = Field(
        ...,
        description="Examples: Mobile, Tablet, Laptop",
    )

    location: str = Field(
        ...,
        description="Transaction location",
    )

    merchant_category: str = Field(
        ...,
        description=(
            "Examples: Electronics, Clothing, Travel, "
            "Groceries, Restaurants"
        ),
    )

    ip_address_flag: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 means suspicious IP address",
    )

    previous_fraudulent_activity: int = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "1 means previous fraudulent activity exists"
        ),
    )

    daily_transaction_count: int = Field(
        ...,
        ge=0,
        description="Number of transactions during the day",
    )

    avg_transaction_amount_7d: float = Field(
        ...,
        ge=0,
        description=(
            "Average transaction amount "
            "in the previous seven days"
        ),
    )

    failed_transaction_count_7d: int = Field(
        ...,
        ge=0,
        description=(
            "Failed transaction count "
            "in the previous seven days"
        ),
    )

    card_type: str = Field(
        ...,
        description=(
            "Examples: Visa, Mastercard, Amex, Discover"
        ),
    )

    card_age: int = Field(
        ...,
        ge=0,
        description="Card age",
    )

    transaction_distance: float = Field(
        ...,
        ge=0,
        description="Transaction distance",
    )

    authentication_method: str = Field(
        ...,
        description=(
            "Examples: OTP, PIN, Password, Biometric"
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_amount": 325.50,
                "transaction_type": "Online",
                "timestamp": "2023-08-15T22:30:00",
                "account_balance": 12000.00,
                "device_type": "Mobile",
                "location": "New York",
                "merchant_category": "Electronics",
                "ip_address_flag": 1,
                "previous_fraudulent_activity": 1,
                "daily_transaction_count": 12,
                "avg_transaction_amount_7d": 180.00,
                "failed_transaction_count_7d": 3,
                "card_type": "Visa",
                "card_age": 36,
                "transaction_distance": 4800.00,
                "authentication_method": "OTP",
            }
        }
    )


# ============================================================
# 5. API 输出格式
# ============================================================

class GovernancePredictionResponse(BaseModel):
    policy_version: str

    artifact_name: str
    artifact_role: str
    deployment_eligible: bool
    automatic_decision_approved: bool

    rule_triggered: bool
    rule_feature: str
    rule_operator: str
    rule_threshold: int
    observed_rule_value: int

    decision_source: str
    alert_label: str

    diagnostic_probability: float | None
    diagnostic_score: float | None
    diagnostic_model_applied: bool

    recommended_action: str
    governance_warning: str

    derived_time_features: dict[str, int]


# ============================================================
# 6. 输入转换函数
# ============================================================

def build_diagnostic_input(
    transaction: TransactionRequest,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    构建残余诊断模型输入。

    注意：
    - year 被训练脚本排除，因为数据只有 2023 年；
    - failed_transaction_count_7d 仅用于规则层；
    - risk_score 不由 API 接收，也不进入模型。
    """

    timestamp = pd.Timestamp(
        transaction.timestamp
    )

    time_features = {
        "month": int(timestamp.month),
        "day": int(timestamp.day),
        "hour": int(timestamp.hour),
        "day_of_week": int(
            timestamp.dayofweek
        ),
        "is_weekend": int(
            timestamp.dayofweek >= 5
        ),
    }

    transaction_row = {
        "transaction_amount": (
            transaction.transaction_amount
        ),
        "transaction_type": (
            transaction.transaction_type
        ),
        "account_balance": (
            transaction.account_balance
        ),
        "device_type": (
            transaction.device_type
        ),
        "location": transaction.location,
        "merchant_category": (
            transaction.merchant_category
        ),
        "ip_address_flag": (
            transaction.ip_address_flag
        ),
        "previous_fraudulent_activity": (
            transaction.previous_fraudulent_activity
        ),
        "daily_transaction_count": (
            transaction.daily_transaction_count
        ),
        "avg_transaction_amount_7d": (
            transaction.avg_transaction_amount_7d
        ),
        "card_type": transaction.card_type,
        "card_age": transaction.card_age,
        "transaction_distance": (
            transaction.transaction_distance
        ),
        "authentication_method": (
            transaction.authentication_method
        ),
        **time_features,
    }

    input_df = pd.DataFrame(
        [transaction_row]
    )

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required diagnostic features: "
            f"{missing_features}"
        )

    input_df = input_df[
        MODEL_FEATURES
    ]

    return input_df, time_features


# ============================================================
# 7. 首页
# ============================================================

@app.get("/")
def root():
    return {
        "message": (
            "Financial Transaction Risk "
            "Monitoring Governance API"
        ),
        "version": "2.0.0",
        "policy_version": POLICY_VERSION,
        "artifact_role": ARTIFACT_ROLE,
        "deployment_eligible": (
            DEPLOYMENT_ELIGIBLE
        ),
        "health_check": "/health",
        "prediction_endpoint": "/predict",
        "api_documentation": "/docs",
    }


# ============================================================
# 8. 健康检查
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "policy_version": POLICY_VERSION,
        "diagnostic_model_loaded": True,
        "artifact_name": ARTIFACT_NAME,
        "artifact_role": ARTIFACT_ROLE,
        "number_of_diagnostic_features": len(
            MODEL_FEATURES
        ),
        "deployment_eligible": (
            DEPLOYMENT_ELIGIBLE
        ),
        "automatic_decision_approved": (
            AUTOMATIC_DECISION_APPROVED
        ),
        "rule_feature": RULE_FEATURE,
        "rule_operator": ">=",
        "rule_threshold": RULE_THRESHOLD,
        "governance_warning": (
            GOVERNANCE_WARNING
        ),
    }


# ============================================================
# 9. 单笔交易治理评估
# ============================================================

@app.post(
    "/predict",
    response_model=(
        GovernancePredictionResponse
    ),
)
def evaluate_transaction(
    transaction: TransactionRequest,
):
    """
    返回透明规则结果和诊断信息。

    重要：
    - 规则触发只表示 Synthetic Rule Alert；
    - 诊断模型无自动决策权；
    - API 不执行批准、拒绝或交易拦截。
    """

    try:
        observed_rule_value = int(
            getattr(
                transaction,
                RULE_FEATURE,
            )
        )

        rule_triggered = bool(
            observed_rule_value
            >= RULE_THRESHOLD
        )

        input_df, time_features = (
            build_diagnostic_input(
                transaction
            )
        )

        if rule_triggered:
            diagnostic_probability = None
            diagnostic_score = None
            diagnostic_model_applied = False

            decision_source = (
                "synthetic_failed_count_rule"
            )
            alert_label = (
                str(
                    PRIMARY_RULE.get(
                        "trigger_label",
                        "Synthetic Rule Alert",
                    )
                )
            )
            recommended_action = str(
                PRIMARY_RULE.get(
                    "recommended_action",
                    (
                        "Manual review for portfolio "
                        "demonstration only"
                    ),
                )
            )

        else:
            probability = float(
                diagnostic_model.predict_proba(
                    input_df
                )[0, 1]
            )

            diagnostic_probability = round(
                probability,
                6,
            )
            diagnostic_score = round(
                probability * 100,
                2,
            )
            diagnostic_model_applied = True

            decision_source = (
                "residual_diagnostic_model"
            )
            alert_label = (
                "No Reliable Automated Decision"
            )
            recommended_action = (
                "Do not automatically approve, reject, "
                "or block. Diagnostic score is shown for "
                "engineering and model-risk review only."
            )

        logger.info(
            (
                "Evaluation completed | "
                "rule_triggered=%s | "
                "decision_source=%s"
            ),
            rule_triggered,
            decision_source,
        )

        return GovernancePredictionResponse(
            policy_version=POLICY_VERSION,
            artifact_name=ARTIFACT_NAME,
            artifact_role=ARTIFACT_ROLE,
            deployment_eligible=(
                DEPLOYMENT_ELIGIBLE
            ),
            automatic_decision_approved=(
                AUTOMATIC_DECISION_APPROVED
            ),
            rule_triggered=rule_triggered,
            rule_feature=RULE_FEATURE,
            rule_operator=">=",
            rule_threshold=RULE_THRESHOLD,
            observed_rule_value=(
                observed_rule_value
            ),
            decision_source=decision_source,
            alert_label=alert_label,
            diagnostic_probability=(
                diagnostic_probability
            ),
            diagnostic_score=(
                diagnostic_score
            ),
            diagnostic_model_applied=(
                diagnostic_model_applied
            ),
            recommended_action=(
                recommended_action
            ),
            governance_warning=(
                GOVERNANCE_WARNING
            ),
            derived_time_features=(
                time_features
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Transaction evaluation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Transaction evaluation failed: "
                f"{str(exc)}"
            ),
        ) from exc
