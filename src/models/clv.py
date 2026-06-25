"""Customer Lifetime Value via BG/NBD + Gamma-Gamma.

The two-model approach (Fader & Hardie) is the industry standard for
non-contractual, repeat-purchase businesses:
  - BG/NBD models how many purchases a customer will make in the future
  - Gamma-Gamma models the average value of those purchases
  - Multiplying them gives the expected revenue over the prediction
    horizon (default: 365 days)

This module wraps the lifetimes library and degrades gracefully to a
heuristic estimator when the library is missing (e.g. smoke tests).
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from src.features.rfm import build_rfm_table
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter  # type: ignore
    HAS_LIFETIMES = True
except Exception:
    HAS_LIFETIMES = False


@dataclass
class ClvResult:
    horizon_days: int
    clv_col: str = "clv_365d"
    fitted: bool = False
    summary: dict | None = None


def _build_summary(df, col):
    return {
        "mean": round(float(df[col].mean()), 2),
        "median": round(float(df[col].median()), 2),
        "p90": round(float(df[col].quantile(0.9)), 2),
        "p99": round(float(df[col].quantile(0.99)), 2),
        "max": round(float(df[col].max()), 2),
    }


def _avg_revenue_per_customer(df):
    return df.groupby("CustomerID").apply(
        lambda g: g["TotalPrice"].sum() / max(g["InvoiceNo"].nunique(), 1)
    )


def fit_clv(df, horizon_days=365, penalizer=0.01, time_unit_days=30):
    """Return (rfm_with_clv, ClvResult)."""
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = build_rfm_table(df, snapshot_date=snapshot_date)
    rfm["frequency"] = (rfm["Frequency"] - 1).clip(lower=0)
    first_seen = df.groupby("CustomerID")["InvoiceDate"].min()
    rfm["T"] = rfm.index.map(
        lambda cid: (snapshot_date - first_seen.loc[cid]).days / time_unit_days
    )
    rfm["recency"] = (rfm["Recency"] / time_unit_days)
    # lifetimes requires recency == 0 whenever frequency == 0
    rfm.loc[rfm["frequency"] == 0, "recency"] = 0.0
    avg_rev = _avg_revenue_per_customer(df)
    rfm["monetary_value"] = rfm.index.map(avg_rev)
    col = "clv_%dd" % horizon_days

    if not HAS_LIFETIMES:
        est_future_freq = (rfm["Frequency"] * (horizon_days / 365.0)).clip(lower=0)
        rfm[col] = rfm["monetary_value"] * est_future_freq
        return rfm, ClvResult(
            horizon_days=horizon_days, clv_col=col, fitted=False,
            summary=_build_summary(rfm, col),
        )
    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    # GammaGamma also requires frequency > 0; subset on both filters
    pos = rfm[(rfm["monetary_value"] > 0) & (rfm["frequency"] > 0)]
    if len(pos) > 10:
        ggf = GammaGammaFitter(penalizer_coef=penalizer)
        ggf.fit(pos["frequency"], pos["monetary_value"])
        expected = ggf.conditional_expected_average_profit(
            pos["frequency"], pos["monetary_value"]
        )
        rfm.loc[pos.index, "expected_avg_value"] = expected
    else:
        rfm["expected_avg_value"] = rfm["monetary_value"]
    t = horizon_days / time_unit_days
    predicted = bgf.predict(t, rfm["frequency"], rfm["recency"], rfm["T"])
    rfm[col] = (predicted * rfm["expected_avg_value"]).clip(lower=0)
    return rfm, ClvResult(
        horizon_days=horizon_days, clv_col=col, fitted=True,
        summary=_build_summary(rfm, col),
    )


