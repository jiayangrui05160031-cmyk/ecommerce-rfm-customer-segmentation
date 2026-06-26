"""Base interface for data sources + entity-event contract.

Every data source must implement load() returning a Dataset object. The
Dataset contains a transactions DataFrame conforming to the canonical
schema below (the "retail" profile defaults), which is what the rest of
the pipeline consumes:

- CustomerID: int (unique customer identifier)
- InvoiceNo:  str (unique invoice / order identifier)
- InvoiceDate: datetime (timestamp of the transaction)
- StockCode: str (SKU / product identifier)
- Description: str (product description)
- Quantity: int (units purchased, positive)
- UnitPrice: float (price per unit)
- TotalPrice: float (Quantity * UnitPrice)
- Country: str (customer country, optional)

Non-retail data sources (donations, SaaS logins, content engagement,
etc.) can plug in by setting Dataset.mapping + Dataset.profile instead
of reshaping their DataFrame to the retail schema.

The two-level contract:
  - SchemaMapping: which column plays which role (entity / event / time / value)
  - DomainProfile: domain-specific semantics (return predicate, what
    features to enable, display label for "Monetary" in reports)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Iterable
import pandas as pd


# ---------------------------------------------------------------------------
# Contract objects
# ---------------------------------------------------------------------------


@dataclass
class SchemaMapping:
    """Declares which DataFrame column plays which analytical role.

    Required: entity_id, event_id, timestamp, value.
    Optional: item_id (basket), quantity, country, extra categorical
    columns the source wants to expose.
    """

    entity_id: str
    event_id: str
    timestamp: str
    value: str
    item_id: str | None = None
    quantity: str | None = None
    country: str | None = None

    def required(self) -> list[str]:
        return [self.entity_id, self.event_id, self.timestamp, self.value]


@dataclass
class DomainProfile:
    """Domain semantics + feature toggles.

    `is_return` is the canonical example of domain logic that does not
    belong in generic code: retail-II encodes a return as an InvoiceNo
    starting with "C", but donations / SaaS logins / content engagement
    have no such concept. Injecting a callable lets the same downstream
    feature code serve any domain.
    """

    name: str = "retail"
    value_label: str = "Monetary"
    is_return: Callable[[pd.DataFrame, "SchemaMapping"], pd.Series] | None = None
    enable_clv: bool = True
    enable_basket: bool = True
    enable_churn: bool = True
    enable_cohort: bool = True
    enable_forecast: bool = True
    extra_event_filters: Callable[[pd.DataFrame, "SchemaMapping"], pd.Series] | None = None


# ---------------------------------------------------------------------------
# Default factories (retail profile, used by every legacy data source)
# ---------------------------------------------------------------------------


def retail_mapping() -> SchemaMapping:
    """Canonical retail mapping — matches the Online Retail II / Olist shape."""
    return SchemaMapping(
        entity_id="CustomerID",
        event_id="InvoiceNo",
        timestamp="InvoiceDate",
        value="TotalPrice",
        item_id="StockCode",
        quantity="Quantity",
        country="Country",
    )


def retail_profile() -> DomainProfile:
    """Profile used by every retail source. Centralises the 'C' return rule."""
    return DomainProfile(
        name="retail",
        value_label="Monetary",
        is_return=lambda df, m: df[m.event_id].astype(str).str.startswith("C"),
    )


def donation_profile() -> DomainProfile:
    """Donation-domain profile: no returns, no baskets, CLV still useful."""
    return DomainProfile(
        name="donation",
        value_label="TotalDonated",
        is_return=None,
        enable_clv=True,
        enable_basket=False,
        enable_churn=True,
    )


# Registry of named profiles so users can write analyze(df, mapping, "retail")
PROFILES: dict[str, DomainProfile] = {
    "retail": retail_profile(),
    "donation": donation_profile(),
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    name: str
    transactions: pd.DataFrame
    meta: dict
    # NEW: optional contract. When None, downstream code falls back to the
    # retail defaults (back-compat for every existing source).
    mapping: SchemaMapping | None = None
    profile: DomainProfile | None = None

    @property
    def n_rows(self):
        return len(self.transactions)

    @property
    def n_customers(self):
        if self.mapping is not None:
            return int(self.transactions[self.mapping.entity_id].nunique())
        return int(self.transactions["CustomerID"].nunique())

    def resolved_mapping(self) -> SchemaMapping:
        return self.mapping or retail_mapping()

    def resolved_profile(self) -> DomainProfile:
        return self.profile or retail_profile()


# ---------------------------------------------------------------------------
# Source registry (unchanged API)
# ---------------------------------------------------------------------------


class BaseDataSource(ABC):
    name: ClassVar[str] = "base"

    @abstractmethod
    def load(self):
        ...


_REGISTRY = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def load_dataset(name=None):
    """Factory: returns a Dataset by name. Defaults to config.data.source."""
    from src.config import get_config
    if name is None:
        name = get_config().data.source
    if name not in _REGISTRY:
        raise ValueError(
            "Unknown data source '%s'. Available: %s" % (name, sorted(_REGISTRY))
        )
    return _REGISTRY[name]().load()


def available_sources():
    return sorted(_REGISTRY)


# Backwards-compatible alias (kept — see comment in retail_ii.py)
CORE_COLUMNS = [
    "CustomerID", "InvoiceNo", "InvoiceDate",
    "StockCode", "Description",
    "Quantity", "UnitPrice", "TotalPrice", "Country",
]


# ---------------------------------------------------------------------------
# profile_inference — best-effort auto-detection of the mapping
# ---------------------------------------------------------------------------


def profile_inference(
    df: pd.DataFrame,
    *,
    timestamp_hints: Iterable[str] = ("date", "time", "timestamp", "datetime", "_at"),
) -> tuple[SchemaMapping, dict]:
    """Heuristically guess a SchemaMapping for an unknown DataFrame.

    Strategy:
      - entity_id  : column with the highest cardinality ratio
                     (unique / len) below 0.5 — too unique ⇒ probably an event id
      - event_id   : next most unique column that's not a timestamp
      - timestamp  : column whose dtype is datetime OR whose name contains a hint
      - value      : numeric column with the largest mean
      - item_id    : low-cardinality string column with 10..500 unique values
      - quantity   : column whose name matches "qty|quantity|count|units"
      - country    : column whose name matches "country|region|state|geo"

    Returns (SchemaMapping, debug_dict). The debug_dict lists which column
    was picked for each role and the score that won, so the caller can
    show "here is what we guessed — confirm?" instead of silently guessing.
    """
    debug: dict = {}
    n = len(df)
    if n == 0:
        raise ValueError("profile_inference: empty DataFrame")

    # 1. timestamp
    ts_col = None
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            ts_col = c
            debug["timestamp_reason"] = "datetime dtype"
            break
    if ts_col is None:
        lowered = {c.lower(): c for c in df.columns}
        for hint in timestamp_hints:
            for lc, orig in lowered.items():
                if hint in lc:
                    ts_col = orig
                    debug["timestamp_reason"] = "name hint: %r" % hint
                    break
            if ts_col:
                break
    if ts_col is None:
        raise ValueError(
            "profile_inference: no datetime column found. Pass an explicit SchemaMapping."
        )

    # 2. entity vs event id
    # Heuristic: entity = column with the *lowest* unique ratio (still
    # meaningful: < 0.5). event = the column with the *highest* unique
    # ratio that is not the entity. Numeric-id columns tie-break by
    # name (those whose name ends in "id" / "no" win for event).
    candidates = [c for c in df.columns if c != ts_col]
    ratios = []
    for c in candidates:
        try:
            r = df[c].nunique(dropna=True) / max(n, 1)
        except TypeError:
            continue
        ratios.append((c, r))
    entity_col = None
    if ratios:
        # Entity = column with the LARGEST unique count whose ratio is
        # in (0.005, 0.5) — moderate repeats per value, not near-constant
        # (Country) and not near-unique (DonationID). Ranking by absolute
        # unique count (not by ratio) is the right tie-break: a real
        # entity like DonorID has 150 distinct values, while a category
        # column has 4.
        eligible_entity = [(c, r) for c, r in ratios if 0.005 < r < 0.5]
        if eligible_entity:
            entity_col, er = max(eligible_entity, key=lambda x: int(round(x[1] * n)))
            debug["entity_id"] = entity_col
            debug["entity_id_ratio"] = round(er, 4)
    # event = highest-ratio column that is not entity
    event_candidates = [(c, r) for c, r in ratios if c != entity_col]
    event_col = None
    if event_candidates:
        # Prefer columns whose name hints at an id (no/id/uid/uuid/key)
        id_hinted = [
            (c, r) for c, r in event_candidates
            if any(k in c.lower() for k in ("id", "no", "uid", "uuid", "key"))
        ]
        pool = id_hinted or event_candidates
        event_col, evr = max(pool, key=lambda x: x[1])
        debug["event_id"] = event_col
        debug["event_id_ratio"] = round(evr, 4)
    if entity_col is None or event_col is None:
        raise ValueError(
            "profile_inference: could not separate entity from event id. "
            "Pass an explicit SchemaMapping."
        )

    # 3. value: numeric column with the largest mean that is NOT an
    # id-like column (ids have huge means and are never the value we
    # want to sum). Heuristic: skip columns whose name matches an id
    # pattern; then among the rest, prefer columns whose name hints
    # at money (price/amount/total/revenue/value), else fall back to
    # the largest mean of the remaining.
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in (entity_col, event_col, ts_col)
        and not any(k in c.lower() for k in ("id", "no", "uid", "uuid", "key"))
    ]
    value_col = None
    if numeric_cols:
        # priority: "total" first (it's quantity*unit), then "amount",
        # then "revenue", then "price" / "value"
        priority = ("total", "amount", "revenue", "value", "price", "money", "donat")
        money_hinted = []
        for p in priority:
            for c in numeric_cols:
                if p in c.lower() and c not in money_hinted:
                    money_hinted.append(c)
        if money_hinted:
            value_col = money_hinted[0]
        else:
            value_col = max(numeric_cols, key=lambda c: float(df[c].abs().mean()))
        debug["value_col"] = value_col
        debug["value_mean"] = round(float(df[value_col].abs().mean()), 4)

    # 4. item_id: low-cardinality string column 10..500 unique
    item_col = None
    string_cols = [
        c for c in df.select_dtypes(include=["object", "string", "category"]).columns
        if c not in (entity_col, event_col, ts_col)
    ]
    for c in string_cols:
        u = df[c].nunique(dropna=True)
        if 10 <= u <= min(500, n // 5):
            item_col = c
            debug["item_id"] = c
            debug["item_id_unique"] = int(u)
            break

    # 5. quantity / country by name
    quantity_col = country_col = None
    for c in df.columns:
        lc = c.lower()
        if quantity_col is None and any(k in lc for k in ("qty", "quantity", "count", "units")):
            quantity_col = c
        if country_col is None and any(k in lc for k in ("country", "region", "state", "geo")):
            country_col = c

    mapping = SchemaMapping(
        entity_id=entity_col,
        event_id=event_col,
        timestamp=ts_col,
        value=value_col or entity_col,  # last-resort fallback
        item_id=item_col,
        quantity=quantity_col,
        country=country_col,
    )
    return mapping, debug
