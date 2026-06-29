from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


@pytest.fixture
def valid_payload() -> dict:
    """返回一笔符合 API schema 的测试交易。"""
    return {
        "transaction_amount": 325.50,
        "transaction_type": "Online",
        "timestamp": "2023-10-19T20:16:00",
        "account_balance": 12000.00,
        "device_type": "Mobile",
        "location": "New York",
        "merchant_category": "Electronics",
        "ip_address_flag": 0,
        "previous_fraudulent_activity": 0,
        "daily_transaction_count": 12,
        "avg_transaction_amount_7d": 180.00,
        "failed_transaction_count_7d": 3,
        "card_type": "Visa",
        "card_age": 36,
        "transaction_distance": 4800.00,
        "authentication_method": "OTP",
    }


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["version"] == "2.0.0"
    assert body["artifact_role"] == "diagnostic_only"
    assert body["deployment_eligible"] is False
    assert body["health_check"] == "/health"
    assert body["prediction_endpoint"] == "/predict"


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["diagnostic_model_loaded"] is True
    assert body["artifact_role"] == "diagnostic_only"
    assert body["deployment_eligible"] is False
    assert body["automatic_decision_approved"] is False
    assert body["rule_feature"] == "failed_transaction_count_7d"
    assert body["rule_operator"] == ">="
    assert body["rule_threshold"] == 4
    assert body["number_of_diagnostic_features"] == 19


def test_predict_residual_diagnostic_path(
    valid_payload: dict,
) -> None:
    payload = deepcopy(valid_payload)
    payload["failed_transaction_count_7d"] = 3

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["rule_triggered"] is False
    assert body["decision_source"] == "residual_diagnostic_model"
    assert body["diagnostic_model_applied"] is True
    assert body["diagnostic_probability"] is not None
    assert 0.0 <= body["diagnostic_probability"] <= 1.0
    assert body["diagnostic_score"] is not None
    assert 0.0 <= body["diagnostic_score"] <= 100.0

    assert body["deployment_eligible"] is False
    assert body["automatic_decision_approved"] is False
    assert body["alert_label"] == "No Reliable Automated Decision"

    assert "Do not automatically approve" in body[
        "recommended_action"
    ]

    assert body["observed_rule_value"] == 3
    assert body["derived_time_features"] == {
        "month": 10,
        "day": 19,
        "hour": 20,
        "day_of_week": 3,
        "is_weekend": 0,
    }


def test_predict_synthetic_rule_path(
    valid_payload: dict,
) -> None:
    payload = deepcopy(valid_payload)
    payload["failed_transaction_count_7d"] = 4

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["rule_triggered"] is True
    assert body["decision_source"] == "synthetic_failed_count_rule"
    assert body["diagnostic_model_applied"] is False
    assert body["diagnostic_probability"] is None
    assert body["diagnostic_score"] is None

    assert body["deployment_eligible"] is False
    assert body["automatic_decision_approved"] is False
    assert body["alert_label"] == "Synthetic Rule Alert"

    assert body["observed_rule_value"] == 4
    assert body["rule_threshold"] == 4
    assert body["rule_operator"] == ">="

    assert (
        body["recommended_action"]
        == "Manual review for portfolio demonstration only"
    )


def test_rule_path_also_triggers_above_threshold(
    valid_payload: dict,
) -> None:
    payload = deepcopy(valid_payload)
    payload["failed_transaction_count_7d"] = 7

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["rule_triggered"] is True


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("transaction_amount", -1),
        ("account_balance", -10),
        ("ip_address_flag", 2),
        ("previous_fraudulent_activity", -1),
        ("daily_transaction_count", -1),
        ("avg_transaction_amount_7d", -1),
        ("failed_transaction_count_7d", -1),
        ("card_age", -1),
        ("transaction_distance", -1),
    ],
)
def test_invalid_numeric_inputs_return_422(
    valid_payload: dict,
    field: str,
    invalid_value: int | float,
) -> None:
    payload = deepcopy(valid_payload)
    payload[field] = invalid_value

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_missing_required_field_returns_422(
    valid_payload: dict,
) -> None:
    payload = deepcopy(valid_payload)
    payload.pop("transaction_amount")

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_timestamp_returns_422(
    valid_payload: dict,
) -> None:
    payload = deepcopy(valid_payload)
    payload["timestamp"] = "not-a-timestamp"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_unknown_categories_are_handled(
    valid_payload: dict,
) -> None:
    """
    OneHotEncoder(handle_unknown='ignore') 应允许新类别，
    这能验证 API 不会因未知类别直接崩溃。
    """
    payload = deepcopy(valid_payload)
    payload["transaction_type"] = "New Transaction Type"
    payload["device_type"] = "Wearable"
    payload["location"] = "Unknown City"
    payload["merchant_category"] = "Other"
    payload["card_type"] = "Other Card"
    payload["authentication_method"] = "Other Method"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["diagnostic_model_applied"] is True
