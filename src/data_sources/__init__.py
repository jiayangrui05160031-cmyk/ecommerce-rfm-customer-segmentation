"""Data source abstraction layer.

Provides a unified interface to load either Online Retail II (legacy),
the Brazilian Olist e-commerce dataset, or a synthetic mock for tests.
Pick via config.yaml data.source or by passing a name explicitly.
"""
from .base import Dataset, BaseDataSource, load_dataset, available_sources, register
from .retail_ii import RetailIISource  # noqa: F401  (registers on import)
from .olist import OlistSource  # noqa: F401
from .mock import MockSource, make_mock  # noqa: F401

__all__ = ["Dataset", "BaseDataSource", "load_dataset", "available_sources", "register", "make_mock"]
