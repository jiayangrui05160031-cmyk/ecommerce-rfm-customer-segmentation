"""Configuration loader.

Loads config.yaml and exposes a typed AppConfig. Every module reads from
here instead of hard-coding paths and knobs. This keeps the pipeline
configurable and easy to test.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class DataConfig:
    source: str = "retail_ii"
    retail_ii_parts_dir: str = "."
    olist_raw_dir: str = "data/raw/olist"
    processed_dir: str = "data/processed"


@dataclass
class ClusteringConfig:
    k_range: list = field(default_factory=lambda: [3, 4, 5, 6, 7])
    primary_k: int = 4
    algorithms: list = field(default_factory=lambda: ["kmeans", "gmm"])
    use_umap: bool = True
    umap_n_components: int = 2
    random_state: int = 42


@dataclass
class ClvConfig:
    prediction_horizon_days: int = 365
    bg_nbd_penalizer: float = 0.01
    time_unit_days: int = 30


@dataclass
class ChurnConfig:
    definition_days: int = 90
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.05


@dataclass
class SurvivalConfig:
    duration_col: str = "T"
    event_col: str = "E"


@dataclass
class MiningConfig:
    cohort_period: str = "M"
    market_basket_min_support: float = 0.01
    market_basket_min_threshold: float = 1.0
    forecast_horizon_months: int = 6


@dataclass
class LLMConfig:
    provider: str = "mock"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 1024
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    clv: ClvConfig = field(default_factory=ClvConfig)
    churn: ChurnConfig = field(default_factory=ChurnConfig)
    survival: SurvivalConfig = field(default_factory=SurvivalConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    reports_dir: str = "reports"
    images_dir: str = "images"
    random_state: int = 42

    def report_path(self):
        return PROJECT_ROOT / self.reports_dir

    def image_path(self):
        return PROJECT_ROOT / self.images_dir

    def processed_path(self):
        return PROJECT_ROOT / self.data.processed_dir


def _merge(section, target):
    for k, v in section.items():
        if hasattr(target, k):
            setattr(target, k, v)


def _build(cfg):
    app = AppConfig()
    for section in ("data", "clustering", "clv", "churn", "survival", "mining", "llm"):
        if section in cfg:
            _merge(cfg[section], getattr(app, section))
    for k in ("reports_dir", "images_dir", "random_state"):
        if k in cfg:
            setattr(app, k, cfg[k])
    return app


def get_config(path=None):
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    return _build(_load_yaml(Path(path)))


def project_root():
    return PROJECT_ROOT


__all__ = ["get_config", "project_root", "AppConfig"]
