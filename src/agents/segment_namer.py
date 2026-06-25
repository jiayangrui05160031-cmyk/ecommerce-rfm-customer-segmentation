"""Segment Naming Agent.

Reads a cluster profile DataFrame and asks the LLM to give each cluster
a business-language label, marketing priority and one-line tagline.
Falls back to deterministic rule-based naming when the LLM is unavailable.
"""
from __future__ import annotations
import pandas as pd
from src.agents.base import AgentResult, BaseAgent
from src.agents.prompts import render_segment_naming_prompt

RULE_BASED_NAMES = [
    "Champions 冠军 VIP",
    "Loyal Customers 忠诚用户",
    "Potential Loyalists 潜力用户",
    "New Customers 新客",
    "Hibernating 沉睡用户",
    "At Risk 流失风险",
]


def _rule_based_name(profile):
    ordered = profile.sort_values("Monetary", ascending=False)
    names = []
    for i in range(len(ordered)):
        if i < len(RULE_BASED_NAMES):
            names.append(RULE_BASED_NAMES[i])
        else:
            names.append("Segment-%d" % i)
    return names


class SegmentNamingAgent(BaseAgent):
    name = "segment_naming"

    @staticmethod
    def profile_from_rfm(rfm, cluster_col="Cluster"):
        agg = rfm.groupby(cluster_col).agg(
            Recency_mean=("Recency", "mean"),
            Frequency_mean=("Frequency", "mean"),
            Monetary_mean=("Monetary", "mean"),
            customers=("Recency", "size"),
            revenue=("Monetary", "sum"),
        )
        agg["revenue_share_pct"] = (
            agg["revenue"] / agg["revenue"].sum() * 100
        ).round(2)
        if "churn_prob" in rfm.columns:
            ch = rfm.groupby(cluster_col)["churn_prob"].mean().round(3)
            agg = agg.join(ch.rename("churn_rate"))
        return agg.reset_index()

    def run(self, inputs):
        rfm = inputs["rfm"]
        profile = self.profile_from_rfm(rfm)
        prompt = render_segment_naming_prompt(profile.to_dict(orient="records"))
        payload = []
        try:
            text = self._call_llm(prompt)
            data = self._parse_json(text)
            if isinstance(data, dict):
                data = [data]
            if isinstance(data, list) and len(data) == len(profile):
                payload = data
        except Exception:
            pass
        if not payload:
            names = _rule_based_name(profile.rename(columns={
                "Recency_mean": "Recency",
                "Frequency_mean": "Frequency",
                "Monetary_mean": "Monetary",
            }))
            payload = [
                {
                    "cluster_id": int(row["Cluster"]),
                    "business_name": name,
                    "english_name": name,
                    "tagline": "",
                    "priority": "P2",
                    "core_pain": "",
                    "core_desire": "",
                }
                for (_, row), name in zip(profile.iterrows(), names)
            ]
        return AgentResult(
            agent_name=self.name,
            payload={"segments": payload, "profile": profile.to_dict(orient="records")},
            raw_response="",
        )
