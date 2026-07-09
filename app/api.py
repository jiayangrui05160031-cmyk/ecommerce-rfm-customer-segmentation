"""FastAPI service for customer segments, campaign decisions and experiments.

Run locally with:
    uvicorn app.api:app --reload

The demo portfolio is built lazily on the first data request. Health checks
therefore stay instant, while all business endpoints use the same tested RFM
pipeline as the offline report.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.agents import SegmentNamingAgent, StrategyComposerAgent
from src.config import get_config
from src.data_sources import load_dataset
from src.features import add_behavioural_features, build_rfm_table, rfm_score
from src.marketing_experiments import analyze_experiment
from src.models import compare_clusterings, fit_churn, fit_clv


class ExperimentRequest(BaseModel):
    control_visitors: int = Field(gt=0, examples=[5000])
    control_conversions: int = Field(ge=0, examples=[400])
    treatment_visitors: int = Field(gt=0, examples=[5000])
    treatment_conversions: int = Field(ge=0, examples=[500])
    target_audience: int = Field(default=0, ge=0, examples=[100000])
    value_per_conversion: float = Field(default=0, ge=0, examples=[120])
    cost_per_contact: float = Field(default=0, ge=0, examples=[0.5])


def _build_portfolio(source: str = "mock") -> dict:
    cfg = get_config()
    dataset = load_dataset(source)
    rfm = build_rfm_table(dataset.transactions)
    rfm = rfm_score(rfm)
    rfm = add_behavioural_features(dataset.transactions, rfm)
    clusterings, _ = compare_clusterings(rfm, k_range=[3, 4])
    best = max(clusterings, key=lambda result: result.composite_score)
    rfm["Cluster"] = best.labels
    clv, clv_result = fit_clv(
        dataset.transactions,
        horizon_days=cfg.clv.prediction_horizon_days,
    )
    rfm[clv_result.clv_col] = clv[clv_result.clv_col]
    if clv_result.clv_col != "clv_365d":
        rfm["clv_365d"] = rfm[clv_result.clv_col]
    rfm, _ = fit_churn(dataset.transactions, rfm)
    segment_payload = SegmentNamingAgent().run({"rfm": rfm}).payload
    nba_payload = StrategyComposerAgent().run(
        {"rfm": rfm, "segments": segment_payload}
    ).payload
    return {
        "source": dataset.name,
        "rfm": rfm,
        "segments": segment_payload,
        "nba": nba_payload["nba"],
        "model": {
            "algorithm": best.algorithm,
            "composite_score": round(float(best.composite_score), 4),
        },
    }


@lru_cache(maxsize=1)
def get_portfolio() -> dict:
    """Return the cached, deterministic demo portfolio."""
    return _build_portfolio()


def create_app() -> FastAPI:
    api = FastAPI(
        title="RFM Customer Intelligence API",
        version="1.0.0",
        description=(
            "可服务化的客户分群、Next-Best-Action 与营销实验决策 API。"
            "默认使用确定性 mock 数据和 MockLLM，无需 API Key。"
        ),
    )

    @api.get("/")
    def root() -> dict:
        return {
            "service": "RFM Customer Intelligence API",
            "docs": "/docs",
            "health": "/health",
        }

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": get_portfolio.cache_info().currsize > 0}

    @api.get("/segments")
    def list_segments() -> dict:
        state = get_portfolio()
        rfm = state["rfm"]
        names = {
            int(item["cluster_id"]): item.get("business_name", "")
            for item in state["segments"].get("segments", [])
        }
        summary = (
            rfm.groupby("Cluster")
            .agg(
                customers=("Recency", "size"),
                avg_recency=("Recency", "mean"),
                avg_frequency=("Frequency", "mean"),
                avg_monetary=("Monetary", "mean"),
                avg_churn_risk=("churn_prob", "mean"),
                total_clv=("clv_365d", "sum"),
            )
            .round(2)
            .reset_index()
        )
        summary["segment_name"] = summary["Cluster"].map(names)
        summary = summary.rename(columns={"Cluster": "cluster_id"})
        return {
            "source": state["source"],
            "model": state["model"],
            "segments": _records(summary),
        }

    @api.get("/customers/{customer_id}")
    def get_customer(customer_id: str) -> dict:
        state = get_portfolio()
        rfm = state["rfm"]
        matches = rfm.index[rfm.index.astype(str) == customer_id]
        if len(matches) == 0:
            raise HTTPException(status_code=404, detail="customer not found")
        row = rfm.loc[matches[0]]
        recommendation = next(
            (
                item
                for item in state["nba"]
                if str(item["customer_id"]) == customer_id
            ),
            None,
        )
        profile = {"customer_id": customer_id, **_jsonable(row.to_dict())}
        return {"profile": profile, "next_best_action": recommendation}

    @api.get("/campaign/recommendations")
    def campaign_recommendations(
        limit: Annotated[int, Query(ge=1, le=1000)] = 50,
        budget: Annotated[float | None, Query(gt=0)] = None,
        channel: str | None = None,
    ) -> dict:
        rows = get_portfolio()["nba"]
        if channel:
            rows = [row for row in rows if row["channel"] == channel]
        selected = []
        spent = 0.0
        for row in rows:
            cost = float(row["cost_per_touch"])
            if budget is not None and spent + cost > budget:
                continue
            selected.append(row)
            spent += cost
            if len(selected) >= limit:
                break
        return {
            "selected_customers": len(selected),
            "estimated_cost": round(spent, 2),
            "expected_incremental_profit": round(
                sum(float(row["expected_incremental_profit"]) for row in selected),
                2,
            ),
            "recommendations": selected,
        }

    @api.post("/experiments/analyze")
    def experiment(request: ExperimentRequest) -> dict:
        if request.control_conversions > request.control_visitors:
            raise HTTPException(422, "control conversions exceed visitors")
        if request.treatment_conversions > request.treatment_visitors:
            raise HTTPException(422, "treatment conversions exceed visitors")
        return analyze_experiment(**request.model_dump()).to_dict()

    return api


def _records(frame: pd.DataFrame) -> list[dict]:
    return [_jsonable(row) for row in frame.to_dict(orient="records")]


def _jsonable(values: dict) -> dict:
    result = {}
    for key, value in values.items():
        if pd.isna(value):
            result[key] = None
        elif hasattr(value, "item"):
            result[key] = value.item()
        else:
            result[key] = value
    return result


app = create_app()
