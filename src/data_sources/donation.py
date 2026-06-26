"""Donation-domain data source.

Demonstrates that the pipeline is not coupled to retail:
this source emits a DataFrame that uses different column names
("DonorID" instead of "CustomerID", "DonationDate" instead of
"InvoiceDate", "Amount" instead of "TotalPrice") and the same
downstream feature code can analyse it.

The point is to prove SchemaMapping + DomainProfile make the
pipeline source-agnostic. The mock generator below produces a
small synthetic donation history with three donor archetypes:
recurring givers, one-off donors, and lapsed donors.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .base import (
    BaseDataSource, Dataset, register,
    SchemaMapping, DomainProfile, donation_profile,
)


# ---------------------------------------------------------------------------
# Schema for the donation domain
# ---------------------------------------------------------------------------


def donation_mapping() -> SchemaMapping:
    """Column roles for the donation-domain source."""
    return SchemaMapping(
        entity_id="DonorID",
        event_id="DonationID",
        timestamp="DonationDate",
        value="Amount",
        country="Country",
        # no item_id, no quantity — donations are scalar amounts
    )


# ---------------------------------------------------------------------------
# Synthetic donor generator
# ---------------------------------------------------------------------------


def make_mock_donations(
    n_donors=150, seed=42,
    start_date="2022-01-01", end_date="2024-12-31",
) -> pd.DataFrame:
    """Generate a small synthetic donation history for tests / demos."""
    rng = np.random.default_rng(seed)
    donor_ids = rng.integers(10000, 99999, size=n_donors)
    archetype = rng.choice(
        ["recurring", "one_off", "lapsed"],
        size=n_donors, p=[0.25, 0.45, 0.30],
    )
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    span_days = (end - start).days
    rows = []
    for did, at in zip(donor_ids, archetype):
        if at == "recurring":
            n = int(rng.integers(5, 14))
            amounts = rng.lognormal(mean=4.5, sigma=0.6, size=n)
            intervals = rng.integers(20, 90, size=n)
        elif at == "one_off":
            n = 1
            amounts = rng.lognormal(mean=4.0, sigma=0.7, size=n)
            intervals = [int(rng.integers(0, span_days))]
        else:  # lapsed — donated early then dropped off
            n = int(rng.integers(1, 4))
            amounts = rng.lognormal(mean=3.8, sigma=0.5, size=n)
            intervals = [int(rng.integers(0, span_days // 2))]
        cursor = start + pd.Timedelta(days=int(intervals[0]))
        for amt, off in zip(amounts, intervals):
            rows.append({
                "DonorID": int(did),
                "DonationID": "D%d" % rng.integers(1_000_000, 9_999_999),
                "DonationDate": cursor,
                "Amount": round(float(amt), 2),
                "Country": rng.choice(["US", "UK", "DE", "BR", "CN"]),
                "Campaign": rng.choice(["emergency", "annual", "monthly", "memorial"]),
            })
            cursor = cursor + pd.Timedelta(days=int(off))
            if cursor > end:
                break
    df = pd.DataFrame(rows).sort_values("DonationDate").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Registered source
# ---------------------------------------------------------------------------


@register
class DonationSource(BaseDataSource):
    name = "donation"

    def __init__(self, n_donors=150, seed=42):
        self.n_donors = n_donors
        self.seed = seed

    def load(self):
        df = make_mock_donations(n_donors=self.n_donors, seed=self.seed)
        return Dataset(
            name=self.name,
            transactions=df,
            meta={"synthetic": True, "n_donors": self.n_donors},
            mapping=donation_mapping(),
            profile=donation_profile(),
        )
