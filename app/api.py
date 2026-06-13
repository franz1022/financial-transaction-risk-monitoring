
from datetime import datetime
from pathlib import Path
import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict


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

# 当前文件位于：
# financial-transaction-risk-monitoring/app/api.py
#
# parents[1] 表示项目根目录：
# financial-transaction-risk-monitoring/
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "fraud_detection_random_forest_pipeline.pkl"
)

FEATURE_FILE = (
    BASE_DIR
    / "models"
    / "model_feature_columns.json"
)

METADATA_FILE = (
    BASE_DIR
    / "models"
    / "model_metadata.json"
)


# ============================================================
# 2. 检查模型文件是否存在
# ============================================================

required_files = [
    MODEL_FILE,
    FEATURE_FILE,
    METADATA_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise RuntimeError(
            f"Required file not found: {file_path}"
        )


# ============================================================
# 3. 加载模型与配置信息
# ============================================================

fraud_model = joblib.load(MODEL_FILE)

feature_info = json.loads(
    FEATURE_FILE.read_text(encoding="utf-8")
)

metadata = json.loads(
    METADATA_FILE.read_text(encoding="utf-8")
)

MODEL_FEATURES = feature_info["feature_cols"]

MODEL_NAME = metadata.get(
    "model_name",
    "Random Forest Fraud Detection Pipeline",
)

LOW_RISK_THRESHOLD = float(
    metadata.get("low_risk_threshold", 0.20)
)

HIGH_RISK_THRESHOLD = float(
    metadata.get("high_risk_threshold", 0.70)
)

logger.info("Fraud model loaded successfully.")
logger.info("Model path: %s", MODEL_FILE)
logger.info("Number of model features: %d", len(MODEL_FEATURES))


# ============================================================
# 4. 创建 FastAPI 应用
# ============================================================

app = FastAPI(
    title="Financial Transaction Risk Monitoring API",
    description=(
        "A machine-learning inference API for transaction-level "
        "fraud probability, risk scoring and recommended actions."
    ),
    version="1.0.0",
)


# ============================================================
# 5. API 输入格式
# ============================================================

class TransactionRequest(BaseModel):
    """
    用户提交一笔新的交易。

    timestamp 会自动拆分为：
    year、month、day、hour、day_of_week、is_weekend。
    """

    transaction_amount: float = Field(
        ...,
        ge=0,
        description="Current transaction amount",
    )

    transaction_type: str = Field(
        ...,
        description=(
            "Examples: Online, POS, ATM Withdrawal, Bank Transfer"
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
        description="1 means previous fraudulent activity exists",
    )

    daily_transaction_count: int = Field(
        ...,
        ge=0,
        description="Number of transactions during the day",
    )

    avg_transaction_amount_7d: float = Field(
        ...,
        ge=0,
        description="Average transaction amount in the last seven days",
    )

    failed_transaction_count_7d: int = Field(
        ...,
        ge=0,
        description="Failed transaction count in the last seven days",
    )

    card_type: str = Field(
        ...,
        description="Examples: Visa, Mastercard, Amex, Discover",
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
        description="Examples: OTP, PIN, Password, Biometric",
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
                "authentication_method": "OTP"
            }
        }
    )


# ============================================================
# 6. API 输出格式
# ============================================================

class FraudPredictionResponse(BaseModel):
    model_name: str

    fraud_probability: float
    model_risk_score: float

    default_binary_prediction: int
    default_prediction_threshold: float

    risk_level: str
    recommended_action: str

    low_risk_threshold: float
    high_risk_threshold: float

    derived_time_features: dict[str, int]


# ============================================================
# 7. 风险等级函数
# ============================================================

def assign_risk_level(
    fraud_probability: float,
) -> tuple[str, str]:
    """
    把 fraud probability 转换成风险等级和业务动作。
    """

    if fraud_probability < LOW_RISK_THRESHOLD:
        return "Low", "Auto Approve"

    if fraud_probability < HIGH_RISK_THRESHOLD:
        return "Medium", "Monitor / Secondary Check"

    return "High", "Manual Review / Alert"


# ============================================================
# 8. 输入数据转换函数
# ============================================================

def build_model_input(
    transaction: TransactionRequest,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    把 API 输入转换成模型训练时使用的 21 个字段。
    """

    timestamp = pd.Timestamp(transaction.timestamp)

    time_features = {
        "year": int(timestamp.year),
        "month": int(timestamp.month),
        "day": int(timestamp.day),
        "hour": int(timestamp.hour),
        "day_of_week": int(timestamp.dayofweek),
        "is_weekend": int(timestamp.dayofweek >= 5),
    }

    transaction_row = {
        "transaction_amount": transaction.transaction_amount,
        "transaction_type": transaction.transaction_type,
        "account_balance": transaction.account_balance,
        "device_type": transaction.device_type,
        "location": transaction.location,
        "merchant_category": transaction.merchant_category,
        "ip_address_flag": transaction.ip_address_flag,
        "previous_fraudulent_activity": (
            transaction.previous_fraudulent_activity
        ),
        "daily_transaction_count": (
            transaction.daily_transaction_count
        ),
        "avg_transaction_amount_7d": (
            transaction.avg_transaction_amount_7d
        ),
        "failed_transaction_count_7d": (
            transaction.failed_transaction_count_7d
        ),
        "card_type": transaction.card_type,
        "card_age": transaction.card_age,
        "transaction_distance": transaction.transaction_distance,
        "authentication_method": (
            transaction.authentication_method
        ),
        **time_features,
    }

    input_df = pd.DataFrame([transaction_row])

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required model features: {missing_features}"
        )

    # 确保输入字段顺序和训练时完全一致
    input_df = input_df[MODEL_FEATURES]

    return input_df, time_features


# ============================================================
# 9. 首页
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Financial Transaction Risk Monitoring API",
        "model_name": MODEL_NAME,
        "health_check": "/health",
        "prediction_endpoint": "/predict",
        "api_documentation": "/docs",
    }


# ============================================================
# 10. 健康检查
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": MODEL_NAME,
        "number_of_features": len(MODEL_FEATURES),
        "low_risk_threshold": LOW_RISK_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
    }


# ============================================================
# 11. 单笔交易预测
# ============================================================

@app.post(
    "/predict",
    response_model=FraudPredictionResponse,
)
def predict_fraud(
    transaction: TransactionRequest,
):
    """
    接收一笔交易，返回：

    1. Fraud Probability
    2. 0–100 Model Risk Score
    3. Default Binary Prediction
    4. Low / Medium / High Risk Level
    5. Recommended Business Action
    """

    try:
        input_df, time_features = build_model_input(transaction)

        fraud_probability = float(
            fraud_model.predict_proba(input_df)[0, 1]
        )

        model_risk_score = fraud_probability * 100

        default_binary_prediction = int(
            fraud_probability >= 0.50
        )

        risk_level, recommended_action = assign_risk_level(
            fraud_probability
        )

        logger.info(
            "Prediction completed | probability=%.4f | risk=%s",
            fraud_probability,
            risk_level,
        )

        return FraudPredictionResponse(
            model_name=MODEL_NAME,
            fraud_probability=round(
                fraud_probability,
                6,
            ),
            model_risk_score=round(
                model_risk_score,
                2,
            ),
            default_binary_prediction=(
                default_binary_prediction
            ),
            default_prediction_threshold=0.50,
            risk_level=risk_level,
            recommended_action=recommended_action,
            low_risk_threshold=LOW_RISK_THRESHOLD,
            high_risk_threshold=HIGH_RISK_THRESHOLD,
            derived_time_features=time_features,
        )

    except Exception as exc:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc

