
from datetime import date, time, datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import os


# ============================================================
# 0. 页面设置
# ============================================================

st.set_page_config(
    page_title="Financial Transaction Risk Monitoring",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# 1. 项目路径和 API 设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"


# ============================================================
# 2. 工具函数
# ============================================================

def check_api_health():
    """
    检查 FastAPI 是否正在运行。
    """

    try:
        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=5,
        )

        if response.status_code == 200:
            return True, response.json()

        return False, {
            "detail": f"API returned status {response.status_code}"
        }

    except requests.RequestException as exc:
        return False, {
            "detail": str(exc)
        }


def request_prediction(transaction_data):
    """
    把交易数据发送给 FastAPI /predict。
    """

    response = requests.post(
        PREDICT_ENDPOINT,
        json=transaction_data,
        timeout=15,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"API request failed: {response.status_code} "
            f"{response.text}"
        )

    return response.json()


@st.cache_data
def load_csv(file_path):
    """
    缓存读取 CSV，避免页面每次操作都重新读取。
    """

    path = Path(file_path)

    if not path.exists():
        return None

    return pd.read_csv(path)


def show_risk_result(result):
    """
    展示 FastAPI 返回的风险预测结果。
    """

    risk_level = result["risk_level"]
    fraud_probability = result["fraud_probability"]
    risk_score = result["model_risk_score"]
    action = result["recommended_action"]

    st.subheader("Prediction Result")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Fraud Probability",
        f"{fraud_probability * 100:.2f}%",
    )

    col2.metric(
        "Model Risk Score",
        f"{risk_score:.2f} / 100",
    )

    col3.metric(
        "Risk Level",
        risk_level,
    )

    col4.metric(
        "Binary Prediction",
        result["default_binary_prediction"],
    )

    if risk_level == "High":
        st.error(
            f"High Risk — Recommended action: {action}"
        )

    elif risk_level == "Medium":
        st.warning(
            f"Medium Risk — Recommended action: {action}"
        )

    else:
        st.success(
            f"Low Risk — Recommended action: {action}"
        )

    st.progress(
        min(max(float(fraud_probability), 0.0), 1.0),
        text=f"Fraud probability: {fraud_probability * 100:.2f}%",
    )

    with st.expander("View derived time features"):
        st.json(result["derived_time_features"])

    with st.expander("View complete API response"):
        st.json(result)


# ============================================================
# 3. 页面标题
# ============================================================

st.title("Financial Transaction Risk Monitoring Dashboard")

st.write(
    """
    This dashboard provides transaction-level fraud prediction,
    model-based risk scoring, risk-level classification and
    operational recommendations through a FastAPI inference service.
    """
)


# ============================================================
# 4. API 状态
# ============================================================

api_ok, api_info = check_api_health()

if api_ok:
    st.success(
        "FastAPI service is online and the fraud model is loaded."
    )

    with st.expander("API and model information"):
        st.json(api_info)

else:
    st.error(
        "FastAPI service is not available. "
        "Please start it before submitting transactions."
    )

    st.code(
        "python -m uvicorn app.api:app --reload",
        language="powershell",
    )

    st.write("API error details:")
    st.json(api_info)


# ============================================================
# 5. 创建 Dashboard Tabs
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Transaction Risk Scoring",
        "Model Performance",
        "Risk Monitoring Overview",
        "High-risk Transactions",
    ]
)


# ============================================================
# Tab 1：单笔交易风险评分
# ============================================================

with tab1:
    st.header("Single Transaction Risk Scoring")

    st.write(
        """
        Enter the raw transaction information below.
        The dashboard sends the transaction to the FastAPI service,
        which automatically derives time features and returns the
        fraud probability, risk score and recommended action.
        """
    )

    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            transaction_amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=325.50,
                step=10.0,
            )

            transaction_type = st.selectbox(
                "Transaction Type",
                [
                    "Online",
                    "POS",
                    "ATM Withdrawal",
                    "Bank Transfer",
                ],
            )

            transaction_date = st.date_input(
                "Transaction Date",
                value=date(2023, 8, 15),
            )

            transaction_time = st.time_input(
                "Transaction Time",
                value=time(22, 30),
            )

            account_balance = st.number_input(
                "Account Balance",
                min_value=0.0,
                value=12000.0,
                step=100.0,
            )

            device_type = st.selectbox(
                "Device Type",
                [
                    "Mobile",
                    "Tablet",
                    "Laptop",
                ],
            )

        with col2:
            location = st.selectbox(
                "Location",
                [
                    "New York",
                    "London",
                    "Tokyo",
                    "Mumbai",
                    "Sydney",
                ],
            )

            merchant_category = st.selectbox(
                "Merchant Category",
                [
                    "Electronics",
                    "Clothing",
                    "Travel",
                    "Groceries",
                    "Restaurants",
                ],
            )

            ip_address_flag = st.selectbox(
                "Suspicious IP Address",
                options=[0, 1],
                format_func=lambda x: (
                    "Yes" if x == 1 else "No"
                ),
            )

            previous_fraudulent_activity = st.selectbox(
                "Previous Fraudulent Activity",
                options=[0, 1],
                format_func=lambda x: (
                    "Yes" if x == 1 else "No"
                ),
            )

            daily_transaction_count = st.number_input(
                "Daily Transaction Count",
                min_value=0,
                value=12,
                step=1,
            )

            avg_transaction_amount_7d = st.number_input(
                "Average Transaction Amount — Last 7 Days",
                min_value=0.0,
                value=180.0,
                step=10.0,
            )

        with col3:
            failed_transaction_count_7d = st.number_input(
                "Failed Transactions — Last 7 Days",
                min_value=0,
                value=3,
                step=1,
            )

            card_type = st.selectbox(
                "Card Type",
                [
                    "Visa",
                    "Mastercard",
                    "Amex",
                    "Discover",
                ],
            )

            card_age = st.number_input(
                "Card Age",
                min_value=0,
                value=36,
                step=1,
            )

            transaction_distance = st.number_input(
                "Transaction Distance",
                min_value=0.0,
                value=4800.0,
                step=100.0,
            )

            authentication_method = st.selectbox(
                "Authentication Method",
                [
                    "OTP",
                    "PIN",
                    "Password",
                    "Biometric",
                ],
            )

        submit_prediction = st.form_submit_button(
            "Evaluate Transaction Risk",
            width="stretch",
        )

    if submit_prediction:
        if not api_ok:
            st.error(
                "Prediction cannot be completed because "
                "the FastAPI service is offline."
            )

        else:
            combined_datetime = datetime.combine(
                transaction_date,
                transaction_time,
            )

            transaction_payload = {
                "transaction_amount": transaction_amount,
                "transaction_type": transaction_type,
                "timestamp": combined_datetime.isoformat(),
                "account_balance": account_balance,
                "device_type": device_type,
                "location": location,
                "merchant_category": merchant_category,
                "ip_address_flag": ip_address_flag,
                "previous_fraudulent_activity": (
                    previous_fraudulent_activity
                ),
                "daily_transaction_count": (
                    daily_transaction_count
                ),
                "avg_transaction_amount_7d": (
                    avg_transaction_amount_7d
                ),
                "failed_transaction_count_7d": (
                    failed_transaction_count_7d
                ),
                "card_type": card_type,
                "card_age": card_age,
                "transaction_distance": transaction_distance,
                "authentication_method": authentication_method,
            }

            try:
                with st.spinner(
                    "Sending transaction to the fraud detection API..."
                ):
                    result = request_prediction(
                        transaction_payload
                    )

                show_risk_result(result)

            except Exception as exc:
                st.exception(exc)


# ============================================================
# Tab 2：模型表现
# ============================================================

with tab2:
    st.header("Supervised Model Performance")

    model_comparison_file = (
        OUTPUT_DIR
        / "class_imbalance_model_comparison.csv"
    )

    model_comparison = load_csv(
        model_comparison_file
    )

    if model_comparison is not None:
        st.subheader("Model Comparison")

        st.dataframe(
            model_comparison,
            width="stretch",
            hide_index=True,
        )

        if (
            "model" in model_comparison.columns
            and "pr_auc" in model_comparison.columns
        ):
            chart_data = (
                model_comparison[
                    ["model", "pr_auc"]
                ]
                .set_index("model")
            )

            st.subheader("PR-AUC by Model")
            st.bar_chart(chart_data)

        if (
            "model" in model_comparison.columns
            and "recall" in model_comparison.columns
        ):
            recall_data = (
                model_comparison[
                    ["model", "recall"]
                ]
                .set_index("model")
            )

            st.subheader("Recall by Model")
            st.bar_chart(recall_data)

    else:
        st.warning(
            "class_imbalance_model_comparison.csv "
            "was not found."
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        figure_path = (
            FIGURE_DIR
            / "18_model_comparison_pr_auc.png"
        )

        if figure_path.exists():
            st.image(
                str(figure_path),
                caption="Model Comparison by PR-AUC",
                width="stretch",
            )

    with col2:
        figure_path = (
            FIGURE_DIR
            / "21_precision_recall_curve_class_imbalance_models.png"
        )

        if figure_path.exists():
            st.image(
                str(figure_path),
                caption="Precision-Recall Curves",
                width="stretch",
            )

    st.subheader("Supervised vs Unsupervised Modeling")

    comparison_file = (
        OUTPUT_DIR
        / "supervised_vs_anomaly_detection_comparison.csv"
    )

    supervised_vs_anomaly = load_csv(
        comparison_file
    )

    if supervised_vs_anomaly is not None:
        st.dataframe(
            supervised_vs_anomaly,
            width="stretch",
            hide_index=True,
        )

        st.info(
            "Supervised Random Forest is the primary fraud model. "
            "Isolation Forest is retained as a complementary "
            "early-warning layer for unusual transactions."
        )


# ============================================================
# Tab 3：风险监控概览
# ============================================================

with tab3:
    st.header("Risk Monitoring Overview")

    risk_summary_file = (
        OUTPUT_DIR
        / "risk_level_summary.csv"
    )

    risk_summary = load_csv(
        risk_summary_file
    )

    if risk_summary is not None:
        st.subheader("Risk-level Summary")

        st.dataframe(
            risk_summary,
            width="stretch",
            hide_index=True,
        )

        total_transactions = int(
            risk_summary["transaction_count"].sum()
        )

        low_row = risk_summary[
            risk_summary["risk_level"] == "Low"
        ]

        medium_row = risk_summary[
            risk_summary["risk_level"] == "Medium"
        ]

        high_row = risk_summary[
            risk_summary["risk_level"] == "High"
        ]

        low_count = (
            int(low_row["transaction_count"].iloc[0])
            if len(low_row) > 0
            else 0
        )

        medium_count = (
            int(medium_row["transaction_count"].iloc[0])
            if len(medium_row) > 0
            else 0
        )

        high_count = (
            int(high_row["transaction_count"].iloc[0])
            if len(high_row) > 0
            else 0
        )

        high_fraud_rate = (
            float(high_row["fraud_rate_pct"].iloc[0])
            if len(high_row) > 0
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Scored Transactions",
            f"{total_transactions:,}",
        )

        col2.metric(
            "Low Risk",
            f"{low_count:,}",
        )

        col3.metric(
            "Medium Risk",
            f"{medium_count:,}",
        )

        col4.metric(
            "High Risk",
            f"{high_count:,}",
            delta=f"Fraud rate {high_fraud_rate:.2f}%",
        )

        chart_data = (
            risk_summary[
                ["risk_level", "transaction_count"]
            ]
            .set_index("risk_level")
        )

        st.subheader("Transaction Distribution by Risk Level")
        st.bar_chart(chart_data)

    else:
        st.warning(
            "risk_level_summary.csv was not found."
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        figure_path = (
            FIGURE_DIR
            / "27_transaction_count_by_risk_level.png"
        )

        if figure_path.exists():
            st.image(
                str(figure_path),
                caption="Transaction Count by Risk Level",
                width="stretch",
            )

    with col2:
        figure_path = (
            FIGURE_DIR
            / "28_fraud_rate_by_risk_level.png"
        )

        if figure_path.exists():
            st.image(
                str(figure_path),
                caption="Fraud Rate by Risk Level",
                width="stretch",
            )

    figure_path = (
        FIGURE_DIR
        / "29_model_risk_score_distribution.png"
    )

    if figure_path.exists():
        st.image(
            str(figure_path),
            caption="Model-based Risk Score Distribution",
            width="stretch",
        )


# ============================================================
# Tab 4：高风险交易清单
# ============================================================

with tab4:
    st.header("High-risk Transaction Monitoring")

    high_risk_file = (
        OUTPUT_DIR
        / "top_100_high_risk_transactions.csv"
    )

    high_risk_df = load_csv(
        high_risk_file
    )

    if high_risk_df is None:
        st.warning(
            "top_100_high_risk_transactions.csv "
            "was not found."
        )

    else:
        st.write(
            """
            This table contains the 100 transactions with the
            highest model-based risk scores in the test dataset.
            """
        )

        display_columns = [
            col
            for col in [
                "scored_transaction_id",
                "transaction_amount",
                "transaction_type",
                "device_type",
                "location",
                "merchant_category",
                "ip_address_flag",
                "previous_fraudulent_activity",
                "failed_transaction_count_7d",
                "transaction_distance",
                "random_forest_fraud_probability",
                "model_risk_score",
                "risk_level",
                "recommended_action",
                "actual_fraud_label",
            ]
            if col in high_risk_df.columns
        ]

        st.dataframe(
            high_risk_df[display_columns],
            width="stretch",
            hide_index=True,
        )

        csv_data = high_risk_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download High-risk Transaction List",
            data=csv_data,
            file_name="top_100_high_risk_transactions.csv",
            mime="text/csv",
        )


# ============================================================
# 6. 页脚
# ============================================================

st.divider()

st.caption(
    "Financial Transaction Risk Monitoring System | "
    "Python · SQL · Scikit-learn · XGBoost · "
    "Isolation Forest · FastAPI · Streamlit"
)

