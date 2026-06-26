"""Unified pipeline orchestrator.

run_pipeline() is the single source of truth for "given a Dataset, run
the analysis". Both run_modern.py (CLI) and src.analyze() (library entry)
call into it, so changes to the pipeline logic happen in exactly one
place.

The pipeline runs steps in order, skipping those that are absent from
`steps` OR whose profile.enable_* flag is False. Each step gets the
canonical intermediate state (rfm, clv_res, churn_res, rules, cohort,
fcst, etc.) passed forward, so later steps can read earlier outputs.

The customer_events precomputed aggregation on the Dataset is the
single full-table groupby; downstream steps derive everything from
that one pass.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config, project_root
from src.data_sources import (
    Dataset, SchemaMapping, DomainProfile, retail_mapping, retail_profile,
)
from src.features import build_rfm_table, rfm_score, add_behavioural_features
from src.models import compare_clusterings, fit_clv, fit_churn
from src.mining import mine_association_rules, build_cohort_matrix, forecast_revenue
from src.agents import SegmentNamingAgent, StrategyComposerAgent, ChatAgent
from src.reports import build_html_report


ALL_STEPS = (
    "load", "features", "cluster", "clv", "churn",
    "basket", "cohort", "forecast", "agents", "report",
)


@dataclass
class PipelineResult:
    """Container for everything the pipeline computed."""
    dataset: Dataset
    rfm: pd.DataFrame
    cluster: Any = None
    clv: Any = None
    churn: Any = None
    rules: pd.DataFrame | None = None
    cohort: dict | None = None
    forecast: dict | None = None
    segments: dict = field(default_factory=dict)
    nba: dict = field(default_factory=lambda: {"nba": []})
    chat_examples: list = field(default_factory=list)
    report_path: str | None = None


def precompute_customer_events(ds: Dataset) -> pd.DataFrame:
    """One full-table groupby. Downstream reads columns from this.

    RFM, CLV summary, churn labels, and the IPI helper all read from
    this single aggregation. Saves 3-4 redundant groupbys on a 1M-row
    dataset.
    """
    m = ds.resolved_mapping()
    df = ds.transactions
    agg = df.groupby(m.entity_id).agg(
        first_ts=(m.timestamp, "min"),
        last_ts=(m.timestamp, "max"),
        n_events=(m.event_id, "nunique"),
        total_value=(m.value, "sum"),
    )
    if m.quantity and m.quantity in df.columns:
        agg["total_qty"] = df.groupby(m.entity_id)[m.quantity].sum()
    return agg


def _resolve_col(m: SchemaMapping, default_col: str, role: str) -> str:
    """Map role to actual column name; fall back to retail default."""
    return {
        "entity_id": m.entity_id or "CustomerID",
        "event_id": m.event_id or "InvoiceNo",
        "timestamp": m.timestamp or "InvoiceDate",
        "value": m.value or "TotalPrice",
        "item_id": m.item_id or "StockCode",
        "quantity": m.quantity or "Quantity",
    }[role]


def _coerce_legacy_columns(df: pd.DataFrame, m: SchemaMapping) -> pd.DataFrame:
    """If profile uses retail column names already, pass through."""
    target_cols = {
        "CustomerID": m.entity_id,
        "InvoiceNo": m.event_id,
        "InvoiceDate": m.timestamp,
        "TotalPrice": m.value,
        "StockCode": m.item_id,
        "Quantity": m.quantity,
    }
    df = df.copy()
    for canonical, source in target_cols.items():
        if source and source != canonical and source in df.columns and canonical not in df.columns:
            df[canonical] = df[source]
    # Coerce dtypes that downstream expects
    if "CustomerID" in df.columns:
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    if "TotalPrice" in df.columns:
        df["TotalPrice"] = pd.to_numeric(df["TotalPrice"], errors="coerce")
    return df


def run_pipeline(
    ds: Dataset,
    steps: list[str] | None = None,
    cfg=None,
    out_dir: Path | None = None,
    nba_out: Path | None = None,
    skip_agents_for_speed: bool = False,
) -> PipelineResult:
    """Execute the analytics pipeline on a Dataset.

    Parameters
    ----------
    ds : Dataset
        Source data with mapping/profile.
    steps : list[str], optional
        Subset of ALL_STEPS to run. None means run all enabled steps.
    cfg : AppConfig, optional
        Defaults to get_config().
    out_dir : Path, optional
        Where to write the HTML report (defaults to reports/).
    nba_out : Path, optional
        Where to write the NBA CSV. Skipped when None.
    skip_agents_for_speed : bool
        Skip LLM agents (useful for unit tests / CI).
    """
    cfg = cfg or get_config()
    p = ds.resolved_profile()
    m = ds.resolved_mapping()
    if steps is None:
        steps = list(ALL_STEPS)
    steps = set(steps)

    # Pre-aggregate once.
    events = precompute_customer_events(ds) if "load" in steps else None
    # Downstream feature code still speaks the retail column names.
    # For non-retail profiles we project to those names on the fly.
    df_for_legacy = _coerce_legacy_columns(ds.transactions, m)

    res = PipelineResult(dataset=ds, rfm=pd.DataFrame())

    # 1. Features (RFM + behavioural)
    if "features" in steps:
        rfm = build_rfm_table(df_for_legacy, mapping=m)
        rfm = rfm_score(rfm)
        rfm = add_behavioural_features(
            df_for_legacy, rfm, mapping=m, profile=p,
        )
        res.rfm = rfm

    # 2. Clustering
    if "cluster" in steps and "features" in steps:
        results, _summary = compare_clusterings(
            res.rfm,
            k_range=cfg.clustering.k_range,
            algorithms=cfg.clustering.algorithms,
            random_state=cfg.clustering.random_state,
        )
        best = max(results, key=lambda r: r.composite_score)
        res.rfm["Cluster"] = best.labels
        res.cluster = best

    # 3. CLV
    if "clv" in steps and p.enable_clv and "features" in steps:
        rfm_clv, clv_res = fit_clv(
            df_for_legacy,
            horizon_days=cfg.clv.prediction_horizon_days,
            penalizer=cfg.clv.bg_nbd_penalizer,
            time_unit_days=cfg.clv.time_unit_days,
        )
        res.rfm[clv_res.clv_col] = rfm_clv[clv_res.clv_col]
        res.clv = clv_res

    # 4. Churn
    if "churn" in steps and p.enable_churn and "features" in steps:
        res.rfm, churn_res = fit_churn(
            df_for_legacy, res.rfm,
            definition_days=cfg.churn.definition_days,
            random_state=cfg.churn.random_state,
        )
        res.churn = churn_res

    # 5. Market basket (retail-only by default)
    if "basket" in steps and p.enable_basket:
        try:
            res.rules = mine_association_rules(
                df_for_legacy,
                min_support=cfg.mining.market_basket_min_support,
            )
        except Exception as e:  # mlxtend missing on smoke envs
            res.rules = None

    # 6. Cohort
    if "cohort" in steps and p.enable_cohort:
        res.cohort = build_cohort_matrix(
            df_for_legacy, period=cfg.mining.cohort_period,
        )

    # 7. Forecast
    if "forecast" in steps and p.enable_forecast:
        res.forecast = forecast_revenue(
            df_for_legacy, horizon_months=cfg.mining.forecast_horizon_months,
        )

    # 8. Agents (LLM)
    if "agents" in steps and not skip_agents_for_speed:
        seg_agent = SegmentNamingAgent()
        seg_res = seg_agent.run({"rfm": res.rfm})
        res.segments = seg_res.payload
        strat_agent = StrategyComposerAgent()
        nba_res = strat_agent.run({"rfm": res.rfm, "segments": res.segments})
        res.nba = nba_res.payload
        if nba_out:
            nba_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(res.nba["nba"]).to_csv(
                nba_out, index=False, encoding="utf-8-sig",
            )
        chat = ChatAgent()
        chat.bind_state({
            "rfm": res.rfm,
            "segments": res.segments,
            "history": (res.forecast or {}).get("history"),
        })
        for q in [
            "客户 12345 的状态", "Champions 群体表现如何", "最近营收趋势",
        ]:
            try:
                turn = chat.ask(q)
                res.chat_examples.append({"question": q, "answer": turn.answer})
            except Exception:
                res.chat_examples.append({"question": q, "answer": "(no answer)"})

    # 9. HTML report
    if "report" in steps:
        out_path = (
            out_dir if out_dir is not None
            else project_root() / "reports" / "business_report.html"
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clv_payload = res.clv.summary if (res.clv and "clv" in steps) else None
        res.report_path = str(build_html_report(
            rfm=res.rfm,
            segments_payload=res.segments if "agents" in steps else {},
            clv_payload=clv_payload,
            nba_payload=res.nba if "agents" in steps else {"nba": []},
            chat_examples=res.chat_examples,
            dataset_name=ds.name,
            n_transactions=ds.n_rows,
            out_path=out_path,
        ))

    return res
