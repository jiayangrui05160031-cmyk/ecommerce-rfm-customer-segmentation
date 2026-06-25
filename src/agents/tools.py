"""Tools callable by the chat agent.

Each tool is (state, **kwargs) -> ToolResult. Tools read from agent
state and return a textual summary plus optional structured data.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class ToolResult:
    summary: str
    data: dict | None = None


def query_segment(state, segment_name="all", metric="all"):
    """Look up segment profile from agent state."""
    rfm = state.get("rfm")
    segments_payload = state.get("segments", {})
    if rfm is None:
        return ToolResult("(no data loaded)")
    if "Cluster" not in rfm.columns:
        return ToolResult("Cluster column not found in state.rfm")
    agg = rfm.groupby("Cluster").agg(
        n=("Recency", "size"),
        recency=("Recency", "mean"),
        frequency=("Frequency", "mean"),
        monetary=("Monetary", "mean"),
    ).round(2)
    if segment_name != "all":
        match = None
        for s in segments_payload.get("segments", []):
            if segment_name.lower() in str(s.get("business_name", "")).lower():
                match = s.get("cluster_id")
                break
        if match is not None and match in agg.index:
            row = agg.loc[match]
            summary = (
                segment_name + " segment (cluster " + str(match) + "): "
                + str(int(row["n"])) + " customers, "
                + "avg recency " + str(row["recency"]) + " days, "
                + "avg freq " + str(row["frequency"]) + ", "
                + "avg monetary " + str(row["monetary"]) + "."
            )
            return ToolResult(summary, {"cluster_id": int(match), "row": row.to_dict()})
    summary = "All segments: " + str(agg.to_dict())
    return ToolResult(summary, {"per_cluster": agg.reset_index().to_dict(orient="records")})


def query_customer(state, customer_id):
    """Look up a single customer's RFM and CLV/churn."""
    rfm = state.get("rfm")
    if rfm is None:
        return ToolResult("(no data loaded)")
    try:
        if customer_id in rfm.index:
            row = rfm.loc[customer_id]
        else:
            return ToolResult("Customer " + str(customer_id) + " not found.")
    except Exception as exc:
        return ToolResult("Lookup error: " + str(exc))
    cols = ["Recency", "Frequency", "Monetary"]
    if "churn_prob" in row.index:
        cols.append("churn_prob")
    if "clv_365d" in row.index:
        cols.append("clv_365d")
    summary = (
        "Customer " + str(customer_id) + ": "
        + "Recency=" + str(int(row.get("Recency", 0)))
        + " days, Frequency=" + str(int(row.get("Frequency", 0)))
        + ", Monetary=" + str(round(float(row.get("Monetary", 0)), 2))
    )
    if "churn_prob" in row.index:
        summary += ", churn_prob=" + str(round(float(row["churn_prob"]), 3))
    if "clv_365d" in row.index:
        summary += ", clv_365d=" + str(round(float(row["clv_365d"]), 2))
    return ToolResult(summary, {c: row.get(c) for c in cols if c in row.index})


def query_trend(state, metric="revenue", time_range="all"):
    """Return monthly revenue trend."""
    history = state.get("history")
    if history is None or len(history) == 0:
        return ToolResult("No time-series history available in state.")
    s = history.set_index("ds")["y"]
    last = s.iloc[-1]
    avg = s.mean()
    summary = (
        "Revenue trend: last month=" + str(round(float(last), 2))
        + ", mean=" + str(round(float(avg), 2))
        + ", n_months=" + str(len(s))
    )
    return ToolResult(summary, {"last": float(last), "mean": float(avg), "n_months": int(len(s))})


TOOL_REGISTRY = {
    "query_segment": query_segment,
    "query_customer": query_customer,
    "query_trend": query_trend,
}
