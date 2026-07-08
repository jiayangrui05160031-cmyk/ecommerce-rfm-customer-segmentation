"""Strategy Composer Agent (Next-Best-Action).

Inputs: per-customer profile DataFrame with customer_id, segment_name,
rfm_score, churn_prob, clv_12m.
Outputs: per-customer recommended marketing action with channel,
expected ROI and reasoning.

The LLM is called ONCE with a segment-level prompt. It returns a JSON
object: {segment_name: strategy_template}. The agent then instantiates
each customer using their segment's strategy template, scaled by their
own churn_prob and clv_12m.

Falls back to a deterministic rule engine when the LLM is unavailable.
"""
from __future__ import annotations
import pandas as pd
from src.agents.base import AgentResult, BaseAgent
from src.agents.prompts import render_strategy_composer_prompt

SEGMENT_RULES = {
    "Champions": ("VIP 专属 9 折 + 私人客服回访", "email", 0.18, 2.0),
    "Loyal": ("会员日特权 + 积分翻倍", "app_push", 0.15, 2.0),
    "New": ("首单 8 折复购券 + 欢迎邮件", "email", 0.20, 1.5),
    "Hibernating": ("唤醒短信 + 大额折扣", "sms", 0.08, 1.0),
    "At Risk": ("限时 7 折 + 个性化推荐", "email", 0.12, 1.5),
    "Potential Loyalists": ("品类探索券 + 试用装", "app_push", 0.14, 1.5),
}
DEFAULT_RULE = ("标准化邮件触达", "email", 0.05, 2.0)


def _rule_based_template(segment_name):
    s = str(segment_name or "")
    for key, val in SEGMENT_RULES.items():
        if key in s:
            return {
                "recommended_action": val[0],
                "channel": val[1],
                "expected_conversion_rate": val[2],
                "cost_per_touch": val[3],
                "reasoning": "default rule for " + key,
            }
    return {
        "recommended_action": DEFAULT_RULE[0],
        "channel": DEFAULT_RULE[1],
        "expected_conversion_rate": DEFAULT_RULE[2],
        "cost_per_touch": DEFAULT_RULE[3],
        "reasoning": "default rule (no segment match)",
    }


def _instantiate(customer_row, tpl):
    clv = float(customer_row.get("clv_12m", 0) or 0)
    churn = float(customer_row.get("churn_prob", 0.5) or 0.5)
    base_conv = float(tpl.get("expected_conversion_rate", 0.1) or 0.1)
    adj_conv = base_conv * (1.0 - 0.5 * churn)
    revenue = clv * adj_conv
    cost = float(tpl.get("cost_per_touch", 2.0) or 2.0)
    incremental_profit = revenue - cost
    roi = (revenue - cost) / max(cost, 0.01)
    priority = incremental_profit * (1.0 + churn)
    return {
        "customer_id": customer_row["customer_id"],
        "segment_name": str(customer_row.get("segment_name", "")),
        "recommended_action": tpl.get("recommended_action", DEFAULT_RULE[0]),
        "channel": tpl.get("channel", DEFAULT_RULE[1]),
        "expected_conversion_rate": round(adj_conv, 4),
        "expected_revenue_per_customer": round(revenue, 2),
        "cost_per_touch": cost,
        "expected_incremental_profit": round(incremental_profit, 2),
        "expected_roi": round(roi, 2),
        "campaign_priority_score": round(priority, 2),
        "reasoning": tpl.get("reasoning", ""),
    }


class StrategyComposerAgent(BaseAgent):
    name = "strategy_composer"

    @staticmethod
    def _build_segment_summary(rfm, segments_payload):
        seg_names = {
            int(s["cluster_id"]): s.get("business_name", "")
            for s in segments_payload.get("segments", [])
        }
        df = rfm.copy()
        if "Cluster" not in df.columns:
            return []
        df["segment_name"] = df["Cluster"].map(seg_names).fillna("")
        num_cols = [c for c in df.columns if c != "segment_name" and df[c].dtype.kind in "fi"]
        agg = df.groupby("segment_name")[num_cols].mean().round(2).reset_index()
        return agg.to_dict(orient="records")

    @staticmethod
    def _build_customer_table(rfm, segments_payload):
        seg_names = {
            int(s["cluster_id"]): s.get("business_name", "")
            for s in segments_payload.get("segments", [])
        }
        df = rfm.reset_index()
        if "customer_id" not in df.columns and "CustomerID" in df.columns:
            df = df.rename(columns={"CustomerID": "customer_id"})
        if "customer_id" not in df.columns:
            df = df.rename(columns={df.columns[0]: "customer_id"})
        df["customer_id"] = df["customer_id"].astype(str)
        df["segment_name"] = (
            df["Cluster"].map(seg_names) if "Cluster" in df.columns else ""
        )
        if "churn_prob" not in df.columns:
            df["churn_prob"] = 0.5
        if "clv_365d" not in df.columns and "clv_12m" not in df.columns:
            df["clv_12m"] = df.get("Monetary", 0) * 0.5
        elif "clv_365d" in df.columns:
            df["clv_12m"] = df["clv_365d"]
        return df

    def run(self, inputs):
        rfm = inputs["rfm"]
        segments = inputs.get("segments", {})
        seg_summary = self._build_segment_summary(rfm, segments)
        templates = {}
        try:
            prompt = render_strategy_composer_prompt(seg_summary)
            text = self._call_llm(prompt)
            data = self._parse_json(text)
            if isinstance(data, dict):
                templates = data
        except Exception:
            pass
        for row in seg_summary:
            seg_name = row.get("segment_name", "")
            if seg_name and seg_name not in templates:
                templates[seg_name] = _rule_based_template(seg_name)
        customers = self._build_customer_table(rfm, segments)
        rows = []
        for _, customer in customers.iterrows():
            seg = str(customer.get("segment_name", ""))
            tpl = templates.get(seg, _rule_based_template(seg))
            rows.append(_instantiate(customer, tpl))
        rows_df = pd.DataFrame(rows)
        if not rows_df.empty:
            rows_df = rows_df.sort_values(
                ["campaign_priority_score", "expected_roi"],
                ascending=[False, False],
            )
        return AgentResult(
            agent_name=self.name,
            payload={
                "nba": rows_df.to_dict(orient="records"),
                "templates": templates,
                "summary": _summarise(rows_df),
            },
        )


def _summarise(df):
    if len(df) == 0:
        return {"n": 0}
    return {
        "n_customers": int(len(df)),
        "expected_total_revenue": round(float(df["expected_revenue_per_customer"].sum()), 2),
        "expected_total_cost": round(float(df["cost_per_touch"].sum()), 2),
        "expected_total_incremental_profit": round(float(df["expected_incremental_profit"].sum()), 2),
        "expected_avg_roi": round(float(df["expected_roi"].mean()), 2),
        "positive_roi_customers": int((df["expected_roi"] > 0).sum()),
        "top_channels": df["channel"].value_counts().head(5).to_dict(),
        "top_segments": df.groupby("segment_name")["expected_revenue_per_customer"].sum().round(2).head(5).to_dict(),
    }
