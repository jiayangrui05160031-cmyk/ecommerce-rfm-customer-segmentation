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
import warnings
import numpy as np
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


def _heuristic_clv(rfm, horizon_days):
    est_future_freq = (rfm["Frequency"] * (horizon_days / 365.0)).clip(lower=0)
    return (rfm["monetary_value"].clip(lower=0) * est_future_freq).clip(lower=0)


def _prepare_lifetimes_frame(df, snapshot_date, time_unit_days):
    rfm = build_rfm_table(df, snapshot_date=snapshot_date)
    first_seen = df.groupby("CustomerID")["InvoiceDate"].min()
    last_seen = df.groupby("CustomerID")["InvoiceDate"].max()
    rfm["frequency"] = (rfm["Frequency"] - 1).clip(lower=0)
    rfm["T"] = rfm.index.map(
        lambda cid: (snapshot_date - first_seen.loc[cid]).days / time_unit_days
    )
    rfm["recency"] = rfm.index.map(
        lambda cid: (last_seen.loc[cid] - first_seen.loc[cid]).days / time_unit_days
    )
    rfm.loc[rfm["frequency"] == 0, "recency"] = 0.0
    avg_rev = _avg_revenue_per_customer(df)
    rfm["monetary_value"] = rfm.index.map(avg_rev)
    return rfm


def fit_clv(df, horizon_days=365, penalizer=0.01, time_unit_days=30):
    """Return (rfm_with_clv, ClvResult)."""
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = _prepare_lifetimes_frame(df, snapshot_date, time_unit_days)
    col = "clv_%dd" % horizon_days

    if not HAS_LIFETIMES:
        rfm[col] = _heuristic_clv(rfm, horizon_days)
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
    rfm["expected_avg_value"] = rfm["expected_avg_value"].fillna(rfm["monetary_value"])
    t = horizon_days / time_unit_days
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predicted = bgf.predict(t, rfm["frequency"], rfm["recency"], rfm["T"])
        predicted = pd.Series(np.asarray(predicted, dtype=float), index=rfm.index)
        if not np.isfinite(predicted).all():
            raise ValueError("non-finite BG/NBD prediction")
        rfm[col] = (predicted * rfm["expected_avg_value"].clip(lower=0)).clip(lower=0)
    except Exception:
        rfm[col] = _heuristic_clv(rfm, horizon_days)
        return rfm, ClvResult(
            horizon_days=horizon_days, clv_col=col, fitted=False,
            summary=_build_summary(rfm, col),
        )
    return rfm, ClvResult(
        horizon_days=horizon_days, clv_col=col, fitted=True,
        summary=_build_summary(rfm, col),
    )


