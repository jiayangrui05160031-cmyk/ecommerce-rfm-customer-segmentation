"""Revenue forecasting via Prophet with graceful fallback.

Returns historical monthly revenue, fitted in-sample forecast, and an
H-month ahead forecast. When Prophet is unavailable we use statsmodels
Holt-Winters.
"""
from __future__ import annotations
import pandas as pd
try:
    from prophet import Prophet  # type: ignore
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    HAS_SM = True
except Exception:
    HAS_SM = False


def _monthly(df, date_col, price_col):
    s = df.set_index(date_col).resample("ME")[price_col].sum().reset_index()
    s.columns = ["ds", "y"]
    return s


def _holt_winters(history, horizon):
    last_date = pd.Timestamp(history["ds"].iloc[-1])
    last_value = float(history["y"].iloc[-1])
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthEnd(1),
        periods=horizon, freq="ME",
    )
    if not HAS_SM or len(history) < 3:
        return pd.DataFrame({"ds": future_dates, "yhat": [last_value] * horizon})
    values = history["y"].astype(float)
    try:
        if len(history) >= 24:
            model = ExponentialSmoothing(
                values, trend="add", seasonal="add", seasonal_periods=12,
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(values, trend="add", seasonal=None).fit(optimized=True)
        pred = model.forecast(horizon)
    except Exception:
        pred = pd.Series([last_value] * horizon)
    return pd.DataFrame({"ds": future_dates, "yhat": pred.values})


def forecast_revenue(df, date_col="InvoiceDate", price_col="TotalPrice", horizon_months=6):
    monthly = _monthly(df, date_col, price_col)
    if HAS_PROPHET and len(monthly) >= 12:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(monthly)
        future = m.make_future_dataframe(periods=horizon_months, freq="ME")
        fcst = m.predict(future)
        fitted = fcst[fcst["ds"] <= monthly["ds"].max()][["ds", "yhat"]]
        forecast = fcst[fcst["ds"] > monthly["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        return {"history": monthly, "fitted": fitted, "forecast": forecast, "method": "prophet"}
    forecast = _holt_winters(monthly, horizon_months)
    return {"history": monthly, "fitted": monthly, "forecast": forecast, "method": "holt_winters"}
