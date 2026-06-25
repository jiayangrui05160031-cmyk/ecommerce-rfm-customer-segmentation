"""Synthetic data source for tests and smoke runs.

Generates a small, deterministic DataFrame that mimics the canonical
transaction schema. Used by tests/smoke_test.py to validate the whole
pipeline in seconds and by dev / CI when real data is unavailable.

The mock contains three engineered cohorts (Champions / Loyal / Hibernating)
so downstream clustering can recover the structure.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseDataSource, Dataset, register


def make_mock(
    n_customers=200,
    seed=42,
    start_date="2011-01-01",
    end_date="2011-12-09",
):
    rng = np.random.default_rng(seed)
    customer_ids = rng.integers(10000, 99999, size=n_customers)
    cohort = rng.choice(
        ["champion", "loyal", "hibernating"],
        size=n_customers,
        p=[0.2, 0.5, 0.3],
    )
    base_aov = rng.lognormal(mean=4.0, sigma=0.7, size=n_customers)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    span = (end - start).days
    rows = []
    for cid, ch, aov in zip(customer_ids, cohort, base_aov):
        if ch == "champion":
            n_orders = int(rng.integers(8, 20))
            recency_bias = rng.beta(2, 5, size=n_orders)
        elif ch == "loyal":
            n_orders = int(rng.integers(3, 8))
            recency_bias = rng.uniform(0, 1, size=n_orders)
        else:
            n_orders = int(rng.integers(1, 3))
            recency_bias = rng.beta(5, 2, size=n_orders)
        for b in recency_bias:
            days_ago = int(b * span)
            qty = int(rng.integers(1, 12))
            unit_price = max(0.5, aov / qty * rng.uniform(0.85, 1.15))
            rows.append({
                "InvoiceNo": "INV%d" % rng.integers(100000, 999999),
                "StockCode": "SKU%d" % rng.integers(1000, 9999),
                "Description": rng.choice(
                    ["Mug", "T-Shirt", "Notebook", "Lamp", "Headphones"]
                ),
                "Quantity": qty,
                "InvoiceDate": end - pd.Timedelta(days=days_ago),
                "UnitPrice": round(unit_price, 2),
                "CustomerID": int(cid),
                "Country": rng.choice(
                    ["United Kingdom", "Germany", "France", "Brazil"]
                ),
            })
    df = pd.DataFrame(rows)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


@register
class MockSource(BaseDataSource):
    name = "mock"

    def __init__(self, n_customers=200, seed=42):
        self.n_customers = n_customers
        self.seed = seed

    def load(self):
        df = make_mock(n_customers=self.n_customers, seed=self.seed)
        return Dataset(name=self.name, transactions=df, meta={
            "synthetic": True,
            "n_customers": self.n_customers,
        })
