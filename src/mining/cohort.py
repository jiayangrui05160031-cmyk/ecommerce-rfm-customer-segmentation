"""Cohort retention analysis.

Groups customers by first-purchase month and computes how many remain
active in subsequent months. Returns retention matrix plus per-cohort
revenue curves.
"""
from __future__ import annotations
import pandas as pd


def build_cohort_matrix(
    df, customer_col="CustomerID", invoice_col="InvoiceNo",
    date_col="InvoiceDate", price_col="TotalPrice", period="M",
):
    df = df.copy()
    df["order_period"] = df[date_col].dt.to_period(period)
    df["cohort"] = (
        df.groupby(customer_col)[date_col]
        .transform("min")
        .dt.to_period(period)
    )
    df["cohort_index"] = (df["order_period"] - df["cohort"]).apply(lambda x: x.n)
    cohort_data = df.groupby(["cohort", "cohort_index"])[customer_col].nunique().reset_index()
    cohort_counts = cohort_data.pivot(index="cohort", columns="cohort_index", values=customer_col)
    cohort_sizes = cohort_counts.iloc[:, 0]
    retention = cohort_counts.divide(cohort_sizes, axis=0).round(4)
    rev = df.groupby(["cohort", "cohort_index"])[price_col].sum().reset_index()
    rev_curve = rev.pivot(index="cohort", columns="cohort_index", values=price_col)
    return {
        "retention_matrix": retention,
        "cohort_sizes": cohort_sizes,
        "revenue_curves": rev_curve,
    }
