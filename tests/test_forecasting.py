import pandas as pd

from src.mining.forecasting import forecast_revenue


def test_short_monthly_history_forecasts_without_two_seasonal_cycles():
    rows = []
    for i in range(12):
        rows.append(
            {
                "InvoiceDate": pd.Timestamp("2025-01-31") + pd.offsets.MonthEnd(i),
                "TotalPrice": 100.0 + i,
            }
        )
    df = pd.DataFrame(rows)

    out = forecast_revenue(df, horizon_months=3)

    assert out["method"] in {"holt_winters", "prophet"}
    assert len(out["forecast"]) == 3
    assert out["forecast"]["yhat"].notna().all()
