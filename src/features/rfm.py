"""Customer-level feature engineering.

Builds:
  - core RFM (Recency / Frequency / Monetary)
  - behavioural signals (inter-purchase interval variance, category breadth,
    average basket size, returns rate, tenure)
  - temporal signals (active months, days_since_last_order)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_rfm_table(
    df, snapshot_date=None, customer_col="CustomerID",
    invoice_col="InvoiceNo", date_col="InvoiceDate", price_col="TotalPrice",
):
    """Aggregate transactions to customer-level RFM table."""
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm = df.groupby(customer_col).agg(
        Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        Frequency=(invoice_col, "nunique"),
        Monetary=(price_col, "sum"),
    )
    return rfm


def rfm_score(rfm, labels=None):
    """Score R/F/M each in 1..5 via quantile binning."""
    labels = labels or [1, 2, 3, 4, 5]
    rfm = rfm.copy()
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=list(reversed(labels)))
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
    df, rfm, customer_col="CustomerID", invoice_col="InvoiceNo",
    date_col="InvoiceDate", price_col="TotalPrice",
    qty_col="Quantity", stock_col="StockCode",
):
    """Append behavioural and temporal features to the RFM table."""
    rfm = rfm.copy()
    basket = df.groupby([customer_col, invoice_col])[price_col].sum()
    basket_stats = basket.groupby(customer_col).agg(
        avg_basket="mean", std_basket="std",
    )
    rfm = rfm.join(basket_stats, how="left")
    sku = df.groupby(customer_col)[stock_col].nunique().rename("category_breadth")
    rfm = rfm.join(sku, how="left")
    months = df.assign(_m=df[date_col].dt.to_period("M")).groupby(
        customer_col
    )["_m"].nunique().rename("n_active_months")
    rfm = rfm.join(months, how="left")
    tenure = df.groupby(customer_col)[date_col].agg(
        lambda x: (x.max() - x.min()).days
    ).rename("tenure_days")
    rfm = rfm.join(tenure, how="left")
    is_return = df[invoice_col].astype(str).str.startswith("C")
    return_rate = df.assign(_r=is_return).groupby(customer_col)["_r"].mean().rename("return_rate")
    rfm = rfm.join(return_rate, how="left")

    def _ipi(g):
        if len(g) < 2:
            return np.nan
        diffs = g.sort_values(by=date_col)[date_col].diff().dropna().dt.days
        return float(diffs.mean()) if len(diffs) else np.nan

    ipi = df.groupby(customer_col).apply(_ipi).rename("avg_ipi_days")
    rfm = rfm.join(ipi, how="left")
    return rfm.fillna({
        "avg_basket": 0.0, "std_basket": 0.0,
        "category_breadth": 0, "n_active_months": 0,
        "tenure_days": 0, "return_rate": 0.0,
    })
