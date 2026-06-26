"""Customer-level feature engineering.

Builds:
  - core RFM (Recency / Frequency / Monetary)
  - behavioural signals (inter-purchase interval variance, category breadth,
    average basket size, returns rate, tenure)
  - temporal signals (active months, days_since_last_order)

Generic contract:
  Every function accepts an optional ``mapping`` (SchemaMapping) and
  ``profile`` (DomainProfile). When omitted, the retail defaults are
  used, so legacy call-sites (build_rfm_table(df) etc.) keep working
  unchanged. Domain-specific behaviour — e.g. the "InvoiceNo starts
  with C means return" rule — comes from profile.is_return, NOT from
  hard-coded string checks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.data_sources.base import (
    SchemaMapping, DomainProfile, retail_mapping, retail_profile,
)


# ---------------------------------------------------------------------------
# Internal helpers — read column names from mapping when given
# ---------------------------------------------------------------------------


def _resolve(mapping, profile, customer_col, invoice_col, date_col, price_col,
             qty_col, stock_col):
    """Merge explicit kwargs with mapping/profile (mapping wins on None)."""
    m = mapping or retail_mapping()
    p = profile or retail_profile()
    return {
        "customer_col": customer_col or m.entity_id,
        "invoice_col":  invoice_col  or m.event_id,
        "date_col":     date_col     or m.timestamp,
        "price_col":    price_col    or m.value,
        "qty_col":      qty_col      or m.quantity,
        "stock_col":    stock_col    or m.item_id,
    }, p, m


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_rfm_table(
    df, snapshot_date=None,
    customer_col=None, invoice_col=None, date_col=None, price_col=None,
    mapping: SchemaMapping | None = None, profile: DomainProfile | None = None,
):
    """Aggregate transactions to customer-level RFM table.

    Pass either explicit column names OR a SchemaMapping. Both are
    supported; explicit names win when both are given.
    """
    cols, _p, _m = _resolve(mapping, None, customer_col, invoice_col, date_col, price_col, None, None)
    if snapshot_date is None:
        snapshot_date = df[cols["date_col"]].max() + pd.Timedelta(days=1)
    rfm = df.groupby(cols["customer_col"]).agg(
        Recency=(cols["date_col"], lambda x: (snapshot_date - x.max()).days),
        Frequency=(cols["invoice_col"], "nunique"),
        Monetary=(cols["price_col"], "sum"),
    )
    return rfm


def rfm_score(rfm, labels=None):
    """Score R/F/M each in 1..5 via quantile binning."""
    labels = labels or [1, 2, 3, 4, 5]
    rfm = rfm.copy()
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=labels)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=labels)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=labels)
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    return rfm


def rfm_level(rfm):
    """Bucket RFM total score into 4 levels."""
    total = (
        rfm["R_Score"].astype(int)
        + rfm["F_Score"].astype(int)
        + rfm["M_Score"].astype(int)
    )
    return pd.cut(
        total,
        bins=[0, 6, 9, 12, 15],
        labels=["Low", "Mid", "High", "Champion"],
    ).rename("Customer_Level")


def add_behavioural_features(
    df, rfm,
    customer_col=None, invoice_col=None, date_col=None, price_col=None,
    qty_col=None, stock_col=None,
    mapping: SchemaMapping | None = None, profile: DomainProfile | None = None,
):
    """Append behavioural and temporal features to the RFM table.

    Domain-specific behaviour (return predicate, basket support) is
    read from the profile, not hard-coded.
    """
    cols, p, m = _resolve(mapping, profile, customer_col, invoice_col, date_col, price_col, qty_col, stock_col)
    cc, ic, dc, pc, qc, sc = (cols[k] for k in ("customer_col", "invoice_col", "date_col", "price_col", "qty_col", "stock_col"))
    rfm = rfm.copy()

    # 1. basket stats (only meaningful if profile enables baskets)
    if p.enable_basket and sc and sc in df.columns:
        basket = df.groupby([cc, ic])[pc].sum()
        basket_stats = basket.groupby(cc).agg(avg_basket="mean", std_basket="std")
        rfm = rfm.join(basket_stats, how="left")
    else:
        rfm["avg_basket"] = 0.0
        rfm["std_basket"] = 0.0

    # 2. category breadth (needs item column)
    if sc and sc in df.columns:
        sku = df.groupby(cc)[sc].nunique().rename("category_breadth")
        rfm = rfm.join(sku, how="left")
    else:
        rfm["category_breadth"] = 0

    # 3. active months
    months = df.assign(_m=df[dc].dt.to_period("M")).groupby(cc)["_m"].nunique().rename("n_active_months")
    rfm = rfm.join(months, how="left")

    # 4. tenure
    tenure = df.groupby(cc)[dc].agg(lambda x: (x.max() - x.min()).days).rename("tenure_days")
    rfm = rfm.join(tenure, how="left")

    # 5. return rate — domain-agnostic, comes from profile
    if p.is_return is not None and ic in df.columns:
        is_return = p.is_return(df, m)
        return_rate = (
            df.assign(_r=is_return.astype(int))
            .groupby(cc)["_r"].mean()
            .rename("return_rate")
        )
        rfm = rfm.join(return_rate, how="left")
    else:
        rfm["return_rate"] = 0.0

    # 6. avg inter-purchase interval
    def _ipi(g):
        if len(g) < 2:
            return np.nan
        diffs = g.sort_values(by=dc)[dc].diff().dropna().dt.days
        return float(diffs.mean()) if len(diffs) else np.nan

    ipi = df.groupby(cc).apply(_ipi).rename("avg_ipi_days")
    rfm = rfm.join(ipi, how="left")

    return rfm.fillna({
        "avg_basket": 0.0, "std_basket": 0.0,
        "category_breadth": 0, "n_active_months": 0,
        "tenure_days": 0, "return_rate": 0.0,
    })
