"""Data source abstraction layer.

Provides a unified interface to load either Online Retail II (legacy),
the Brazilian Olist e-commerce dataset, or a synthetic mock for tests.
Pick via config.yaml data.source or by passing a name explicitly.

The Dataset returned by each source carries a SchemaMapping and a
DomainProfile, so downstream feature/model code is source-agnostic
and can serve non-retail domains (donations, SaaS logins) without
reshape.
"""
from .base import (
    Dataset, BaseDataSource,
    load_dataset, available_sources, register,
    SchemaMapping, DomainProfile,
    retail_mapping, retail_profile, donation_profile, PROFILES,
    profile_inference,
)
from .retail_ii import RetailIISource  # noqa: F401  (registers on import)
from .olist import OlistSource  # noqa: F401
from .mock import MockSource, make_mock  # noqa: F401
from .donation import DonationSource, make_mock_donations, donation_mapping  # noqa: F401

__all__ = [
    "Dataset", "BaseDataSource",
    "load_dataset", "available_sources", "register",
    "SchemaMapping", "DomainProfile",
    "retail_mapping", "donation_mapping",
    "retail_profile", "donation_profile", "PROFILES",
    "profile_inference",
    "make_mock_donations",
]
