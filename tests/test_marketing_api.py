"""Unit and contract tests for the customer intelligence API."""
from fastapi.testclient import TestClient
import pytest

from app.api import create_app, get_portfolio
from src.marketing_experiments import analyze_experiment


def test_experiment_detects_profitable_positive_lift():
    result = analyze_experiment(
        control_visitors=5000,
        control_conversions=400,
        treatment_visitors=5000,
        treatment_conversions=520,
        target_audience=100000,
        value_per_conversion=120,
        cost_per_contact=0.5,
    )
    assert result.absolute_lift == pytest.approx(0.024)
    assert result.p_value < 0.05
    assert result.expected_incremental_conversions == pytest.approx(2400)
    assert result.expected_incremental_profit == pytest.approx(238000)
    assert result.recommendation == "scale"


def test_experiment_handles_zero_control_rate():
    result = analyze_experiment(
        control_visitors=100,
        control_conversions=0,
        treatment_visitors=100,
        treatment_conversions=1,
    )
    assert result.relative_lift is None
    assert result.recommendation == "continue_testing"


def test_significant_lift_can_still_fail_unit_economics():
    result = analyze_experiment(
        control_visitors=5000,
        control_conversions=400,
        treatment_visitors=5000,
        treatment_conversions=520,
        target_audience=100000,
        value_per_conversion=1,
        cost_per_contact=1,
    )
    assert result.statistically_significant is True
    assert result.expected_incremental_profit < 0
    assert result.recommendation == "do_not_scale"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"control_visitors": 0, "control_conversions": 0,
         "treatment_visitors": 10, "treatment_conversions": 1},
        {"control_visitors": 10, "control_conversions": 11,
         "treatment_visitors": 10, "treatment_conversions": 1},
    ],
)
def test_experiment_rejects_invalid_samples(kwargs):
    with pytest.raises(ValueError):
        analyze_experiment(**kwargs)


@pytest.fixture(scope="module")
def client():
    get_portfolio.cache_clear()
    return TestClient(create_app())


def test_health_is_fast_and_does_not_eagerly_load_model(client):
    get_portfolio.cache_clear()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_segments_contract(client):
    response = client.get("/segments")
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "mock"
    assert len(payload["segments"]) >= 3
    assert sum(segment["customers"] for segment in payload["segments"]) > 100
    assert all("avg_churn_risk" in segment for segment in payload["segments"])


def test_customer_and_not_found_contract(client):
    customer_id = str(get_portfolio()["rfm"].index[0])
    response = client.get(f"/customers/{customer_id}")
    assert response.status_code == 200
    assert response.json()["next_best_action"]["customer_id"] == customer_id
    assert client.get("/customers/does-not-exist").status_code == 404


def test_budgeted_campaign_never_exceeds_budget(client):
    response = client.get("/campaign/recommendations?limit=100&budget=10")
    assert response.status_code == 200
    payload = response.json()
    assert payload["estimated_cost"] <= 10
    assert payload["selected_customers"] <= 100


def test_experiment_api_contract(client):
    response = client.post(
        "/experiments/analyze",
        json={
            "control_visitors": 5000,
            "control_conversions": 400,
            "treatment_visitors": 5000,
            "treatment_conversions": 520,
            "target_audience": 100000,
            "value_per_conversion": 120,
            "cost_per_contact": 0.5,
        },
    )
    assert response.status_code == 200
    assert response.json()["recommendation"] == "scale"
    invalid = client.post(
        "/experiments/analyze",
        json={
            "control_visitors": 10,
            "control_conversions": 11,
            "treatment_visitors": 10,
            "treatment_conversions": 1,
        },
    )
    assert invalid.status_code == 422
