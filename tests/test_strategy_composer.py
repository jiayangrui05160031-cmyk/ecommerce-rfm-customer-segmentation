from src.agents.strategy_composer import _instantiate


def test_instantiate_adds_incremental_profit_and_priority_score():
    row = {
        "customer_id": 1001,
        "segment_name": "Champions 冠军 VIP",
        "clv_12m": 100.0,
        "churn_prob": 0.2,
    }
    tpl = {
        "recommended_action": "VIP offer",
        "channel": "email",
        "expected_conversion_rate": 0.2,
        "cost_per_touch": 3.0,
        "reasoning": "test",
    }

    out = _instantiate(row, tpl)

    assert out["expected_conversion_rate"] == 0.18
    assert out["expected_revenue_per_customer"] == 18.0
    assert out["expected_incremental_profit"] == 15.0
    assert out["expected_roi"] == 5.0
    assert out["campaign_priority_score"] == 18.0
