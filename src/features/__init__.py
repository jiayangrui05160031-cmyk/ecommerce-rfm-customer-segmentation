"""Feature engineering layer.

Builds customer-level features from raw transactions. Goes beyond the
classic 3-dim RFM to include behavioural and temporal signals.
"""
from .rfm import build_rfm_table, rfm_score, rfm_level, add_behavioural_features

__all__ = [
    "build_rfm_table",
    "rfm_score",
    "rfm_level",
    "add_behavioural_features",
]
