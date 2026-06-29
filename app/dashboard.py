from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
import os

import pandas as pd
import requests
import streamlit as st


# ============================================================
# 0. 页面设置
# ============================================================

st.set_page_config(
    page_title="Financial Transaction Risk Governance",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# 1. 项目路径与 API 设置
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

def check_api_health() -> tuple[bool, dict]:
    """检查 FastAPI 是否在线。"""
    try:
        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=5,
        )

        if response.status_code == 200:
            return True, response.json()

        return False, {
            "detail": (
                "API returned status "
                f"{response.status_code}"
            )
        }

    except requests.RequestException as exc:
        return False, {
            "detail": str(exc)
        }


def request_evaluation(
    transaction_data: dict,
) -> dict:
    """向治理 API 提交单笔交易。"""
    response = requests.post(
        PREDICT_ENDPOINT,
        json=transaction_data,
        timeout=15,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "API request failed: "
            f"{response.status_code} "
            f"{response.text}"
        )

    return response.json()


@st.cache_data
def load_csv(
    file_path: str | Path,
) -> pd.DataFrame | None:
    """缓存读取 CSV。"""
    path = Path(file_path)

    if not path.exists():
        return None

    return pd.read_csv(path)


def display_dataframe_if_exists(
    title: str,
    file_path: Path,
    columns: list[str] | None = None,
) -> None:
    """存在时显示 CSV。"""
    data = load_csv(file_path)

    if data is None:
        st.warning(
            f"{file_path.name} was not found."
        )
        return

    st.subheader(title)

    if columns:
        available_columns = [
            column
            for column in columns
            if column in data.columns
        ]

        if available_columns:
            data = data[available_columns]

    st.dataframe(
        data,
        width="stretch",
        hide_index=True,
    )


def show_governance_result(
    result: dict,
) -> None:
    """展示治理 API 返回结果。"""
    st.subheader("Governance Evaluation Result")

    rule_triggered = bool(
        result["rule_triggered"]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rule Triggered",
        "Yes" if rule_triggered else "No",
    )

    col2.metric(
        "Decision Source",
        result["decision_source"],
    )

    col3.metric(
        "Deployment Eligible",
        (
            "Yes"
            if result["deployment_eligible"]
            else "No"
        ),
    )

    col4.metric(
        "Automated Decision",
        (
            "Approved"
            if result["automatic_decision_approved"]
            else "Not Approved"
        ),
    )

    if rule_triggered:
        st.error(
            f"{result['alert_label']} — "
            f"{result['recommended_action']}"
        )

        st.write(
            f"Observed `{result['rule_feature']}`: "
            f"**{result['observed_rule_value']}**"
        )

        st.write(
            "Transparent rule: "
            f"`{result['rule_feature']} "
            f"{result['rule_operator']} "
            f"{result['rule_threshold']}`"
        )

        st.info(
            "The residual diagnostic model was not applied "
            "because this transaction triggered the "
            "synthetic failed-count rule."
        )

    else:
        st.warning(
            result["alert_label"]
        )

        diagnostic_probability = (
            result.get(
                "diagnostic_probability"
            )
        )

        diagnostic_score = result.get(
            "diagnostic_score"
        )

        metric_col1, metric_col2 = st.columns(2)

        metric_col1.metric(
            "Diagnostic Probability",
            (
                f"{diagnostic_probability * 100:.2f}%"
                if diagnostic_probability is not None
                else "Not available"
            ),
        )

        metric_col2.metric(
            "Diagnostic Score",
            (
                f"{diagnostic_score:.2f} / 100"
                if diagnostic_score is not None
                else "Not available"
            ),
        )

        st.info(
            result["recommended_action"]
        )

        if diagnostic_probability is not None:
            st.progress(
                min(
                    max(
                        float(diagnostic_probability),
                        0.0,
                    ),
                    1.0,
                ),
                text=(
                    "Diagnostic probability only — "
                    "not an approved fraud decision"
                ),
            )

    st.warning(
        result["governance_warning"]
    )

    with st.expander(
        "View derived time features"
    ):
        st.json(
            result["derived_time_features"]
        )

    with st.expander(
        "View complete API response"
    ):
        st.json(result)


# ============================================================
# 3. 页面标题与治理声明
# ============================================================

st.title(
    "Financial Transaction Risk Governance Dashboard"
)

st.write(
    """
    This portfolio dashboard demonstrates transparent rule
    evaluation, residual model diagnostics, out-of-time validation,
    and model-governance controls. It does not approve, reject,
    or block financial transactions.
    """
)

st.warning(
    "Synthetic portfolio dataset only. "
    "No customer-impact decision is approved."
)


# ============================================================
# 4. API 状态
# ============================================================

api_ok, api_info = check_api_health()

if api_ok:
    st.success(
        "FastAPI governance service is online."
    )

    with st.expander(
        "API, policy and artifact information"
    ):
        st.json(api_info)

else:
    st.error(
        "FastAPI is unavailable. Start the API "
        "before submitting a transaction."
    )

    st.code(
        "python -m uvicorn app.api:app --reload",
        language="powershell",
    )

    st.json(api_info)


# ============================================================
# 5. 页面 Tabs
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Transaction Evaluation",
        "Model Validation",
        "Rule & Residual Analysis",
        "Governance Summary",
    ]
)


# ============================================================
# Tab 1：单笔交易治理评估
# ============================================================

with tab1:
    st.header(
        "Single Transaction Governance Evaluation"
    )

    st.write(
        """
        The transparent rule is evaluated first. Transactions
        outside the rule receive a residual diagnostic score,
        but that score has no automated decision authority.
        """
    )

    with st.form(
        "transaction_governance_form"
    ):
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
                value=date(2023, 10, 19),
            )

            transaction_time = st.time_input(
                "Transaction Time",
                value=time(20, 16),
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
                [0, 1],
                format_func=lambda value: (
                    "Yes" if value == 1 else "No"
                ),
            )

            previous_fraudulent_activity = st.selectbox(
                "Previous Fraudulent Activity",
                [0, 1],
                format_func=lambda value: (
                    "Yes" if value == 1 else "No"
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
                help=(
                    "A value of 4 or above triggers "
                    "the transparent synthetic-data rule."
                ),
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

        submit_evaluation = st.form_submit_button(
            "Evaluate Transaction",
            width="stretch",
        )

    if submit_evaluation:
        if not api_ok:
            st.error(
                "Evaluation cannot be completed "
                "because FastAPI is offline."
            )

        else:
            combined_datetime = datetime.combine(
                transaction_date,
                transaction_time,
            )

            transaction_payload = {
                "transaction_amount": (
                    transaction_amount
                ),
                "transaction_type": (
                    transaction_type
                ),
                "timestamp": (
                    combined_datetime.isoformat()
                ),
                "account_balance": (
                    account_balance
                ),
                "device_type": device_type,
                "location": location,
                "merchant_category": (
                    merchant_category
                ),
                "ip_address_flag": (
                    ip_address_flag
                ),
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
                "transaction_distance": (
                    transaction_distance
                ),
                "authentication_method": (
                    authentication_method
                ),
            }

            try:
                with st.spinner(
                    "Sending transaction to "
                    "the governance API..."
                ):
                    result = request_evaluation(
                        transaction_payload
                    )

                show_governance_result(result)

            except Exception as exc:
                st.exception(exc)


# ============================================================
# Tab 2：模型验证证据
# ============================================================

with tab2:
    st.header(
        "Out-of-Time Model Validation"
    )

    st.info(
        "The original high benchmark score was driven "
        "primarily by a deterministic synthetic proxy. "
        "After removing that proxy, the conservative "
        "models performed near random."
    )

    display_dataframe_if_exists(
        "Temporal Feature Ablation — Validation",
        OUTPUT_DIR
        / "temporal_ablation_validation_results.csv",
        [
            "feature_set",
            "model",
            "selection_eligible",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
        ],
    )

    display_dataframe_if_exists(
        "Independent Test — Conservative Candidate",
        OUTPUT_DIR
        / "temporal_ablation_final_test_metrics.csv",
        [
            "feature_set",
            "model",
            "review_rate",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ],
    )

    col1, col2 = st.columns(2)

    with col1:
        figure_path = (
            FIGURE_DIR
            / "37_temporal_ablation_validation_pr_auc.png"
        )

        if figure_path.exists():
            st.image(
                str(figure_path),
                caption=(
                    "Validation PR-AUC: "
                    "benchmark versus conservative features"
                ),
                width="stretch",
            )

    with col2:
        figure_path = (
            FIGURE_DIR
            / "38_temporal_ablation_test_confusion_matrix.png"
        )

        if figure_path.exists():
            st.image(
                str(figure_path),
                caption=(
                    "Independent Test confusion matrix "
                    "for the conservative candidate"
                ),
                width="stretch",
            )


# ============================================================
# Tab 3：规则与残余信号分析
# ============================================================

with tab3:
    st.header(
        "Transparent Rule and Residual Signal"
    )

    display_dataframe_if_exists(
        "Fraud Rate by Failed Transaction Count",
        OUTPUT_DIR
        / "failed_count_fraud_rate_audit.csv",
        [
            "failed_transaction_count_7d",
            "transaction_count",
            "fraud_count",
            "fraud_rate_pct",
        ],
    )

    display_dataframe_if_exists(
        "Residual Model Validation Comparison",
        OUTPUT_DIR
        / "residual_validation_model_comparison.csv",
        [
            "model",
            "fraud_rate",
            "roc_auc",
            "pr_auc",
            "relative_pr_auc_uplift",
        ],
    )

    display_dataframe_if_exists(
        "Two-Layer Test Strategy",
        OUTPUT_DIR
        / "two_layer_test_strategy.csv",
        [
            "scenario",
            "review_rate",
            "precision",
            "recall",
            "f1",
            "residual_incremental_precision",
        ],
    )

    col1, col2 = st.columns(2)

    with col1:
        figure_path = (
            FIGURE_DIR
            / "35_failed_count_fraud_rate_audit.png"
        )

        if figure_path.exists():
            st.image(
                str(figure_path),
                caption=(
                    "Deterministic failed-count pattern"
                ),
                width="stretch",
            )

    with col2:
        figure_path = (
            FIGURE_DIR
            / "40_two_layer_test_strategy.png"
        )

        if figure_path.exists():
            st.image(
                str(figure_path),
                caption=(
                    "Rule-only versus two-layer strategies"
                ),
                width="stretch",
            )

    st.error(
        "Deployment decision: Do not deploy the residual "
        "machine-learning model."
    )


# ============================================================
# Tab 4：治理摘要
# ============================================================

with tab4:
    st.header(
        "Model Governance Summary"
    )

    artifact_summary = load_csv(
        OUTPUT_DIR
        / "diagnostic_model_artifact_summary.csv"
    )

    if artifact_summary is not None:
        row = artifact_summary.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Artifact Role",
            str(row["artifact_role"]),
        )

        col2.metric(
            "Deployment Eligible",
            str(row["deployment_eligible"]),
        )

        col3.metric(
            "Test ROC-AUC",
            f"{float(row['test_roc_auc']):.4f}",
        )

        col4.metric(
            "Test PR-AUC",
            f"{float(row['test_pr_auc']):.4f}",
        )

        st.dataframe(
            artifact_summary,
            width="stretch",
            hide_index=True,
        )

    st.subheader(
        "Approved Uses"
    )

    st.markdown(
        """
        - Engineering demonstration
        - Model-risk diagnostics
        - API and dashboard integration
        - Transparent rule and workload analysis
        """
    )

    st.subheader(
        "Prohibited Uses"
    )

    st.markdown(
        """
        - Automatic transaction approval
        - Automatic rejection
        - Transaction blocking
        - Customer-impact decisions
        """
    )

    st.subheader(
        "Final Positioning"
    )

    st.write(
        """
        This project demonstrates end-to-end risk-monitoring
        engineering together with proxy-feature detection,
        temporal validation, model rejection, transparent rule
        design, and governance-aware system integration.
        """
    )


# ============================================================
# 6. 页脚
# ============================================================

st.divider()

st.caption(
    "Financial Transaction Risk Governance Prototype | "
    "Python · SQL · Scikit-learn · FastAPI · Streamlit · Docker"
)
