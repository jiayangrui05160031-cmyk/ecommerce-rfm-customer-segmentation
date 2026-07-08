import pandas as pd

from src.features.rfm import rfm_score
from src.models.clv import _prepare_lifetimes_frame


def test_recency_score_rewards_more_recent_customers():
    rfm = pd.DataFrame(
        {
            "Recency": [1, 2, 3, 4, 5],
            "Frequency": [1, 2, 3, 4, 5],
            "Monetary": [10, 20, 30, 40, 50],
        }
    )

    scored = rfm_score(rfm)

    assert int(scored.loc[0, "R_Score"]) == 5
    assert int(scored.loc[4, "R_Score"]) == 1


def test_lifetimes_recency_is_first_to_last_purchase_age():
    df = pd.DataFrame(
        {
            "CustomerID": ["a", "a", "b"],
            "InvoiceNo": ["i1", "i2", "i3"],
            "InvoiceDate": pd.to_datetime(["2026-01-01", "2026-01-31", "2026-01-20"]),
            "TotalPrice": [10.0, 20.0, 30.0],
        }
    )
    frame = _prepare_lifetimes_frame(
        df,
        snapshot_date=pd.Timestamp("2026-02-01"),
        time_unit_days=30,
    )

    assert frame.loc["a", "frequency"] == 1
    assert frame.loc["a", "recency"] == 1.0
    assert frame.loc["b", "frequency"] == 0
    assert frame.loc["b", "recency"] == 0.0
