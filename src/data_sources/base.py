"""Base interface for data sources.

Every data source must implement load() returning a Dataset object. The
Dataset contains a transactions DataFrame conforming to the canonical
schema below, which is what the rest of the pipeline consumes:
- CustomerID: int (unique customer identifier)
- InvoiceNo:  str (unique invoice / order identifier)
- InvoiceDate: datetime (timestamp of the transaction)
- StockCode: str (SKU / product identifier)
- Description: str (product description)
- Quantity: int (units purchased, positive)
- UnitPrice: float (price per unit)
- TotalPrice: float (Quantity * UnitPrice)
- Country: str (customer country, optional)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
import pandas as pd


@dataclass
class Dataset:
    name: str
    transactions: pd.DataFrame
    meta: dict

    @property
    def n_rows(self):
        return len(self.transactions)

    @property
    def n_customers(self):
        return int(self.transactions["CustomerID"].nunique())


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


CORE_COLUMNS = [
    "CustomerID", "InvoiceNo", "InvoiceDate",
    "StockCode", "Description",
    "Quantity", "UnitPrice", "TotalPrice", "Country",
]
