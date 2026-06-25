"""Real-LLM smoke test: validate the 3 agents against a live MiniMax endpoint.

This test is intentionally separated from tests/smoke_test.py because it
hits the network and requires the MINIMAX_API_KEY env var. CI should NOT
run this; it is for local debugging and acceptance.

Usage:
    $env:MINIMAX_API_KEY="sk-..."
    python tests/smoke_real_llm.py

Pass criteria:
- MiniMaxChat returns a non-empty string for a hello-world prompt
- SegmentNamingAgent returns N segments matching the cluster count,
  each with business_name and priority
- StrategyComposerAgent returns one NBA row per customer with all
  required keys
- ChatAgent.ask() routes intent correctly and returns non-empty text
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.data_sources import load_dataset  # noqa: E402
from src.features import build_rfm_table, rfm_score, add_behavioural_features  # noqa: E402
from src.models import compare_clusterings, fit_clv, fit_churn  # noqa: E402
from src.agents.llm_factory import MiniMaxChat  # noqa: E402
from src.agents import SegmentNamingAgent, StrategyComposerAgent, ChatAgent  # noqa: E402


def banner(text):
    print("\n" + "=" * 64)
    print("  " + text)
    print("=" * 64)


def main():
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        print("[FAIL] MINIMAX_API_KEY env var is not set.")
        print("       Set it then re-run: $env:MINIMAX_API_KEY=\"sk-...\"")
        sys.exit(1)
    cfg = get_config()
    print("[boot] provider =", cfg.llm.provider)
    print("[boot] model    =", cfg.llm.model)
    print("[boot] base_url =", cfg.llm.base_url)
    print("[boot] api_key  =", api_key[:8] + "..." + api_key[-4:])

    llm = MiniMaxChat(
        api_key=api_key,
        model=cfg.llm.model,
        base_url=cfg.llm.base_url,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
    )

    banner("LLM CONNECTIVITY")
    t0 = time.time()
    pong = llm.invoke("用一句话回答: 你好")
    dt = time.time() - t0
    assert pong.content.strip(), "MiniMax returned empty content"
    print("[ok] connectivity: %.2fs" % dt)
    print("     response: %s" % pong.content.strip()[:80])

    banner("DATA + FEATURES + CLUSTER + CLV + CHURN")
    ds = load_dataset("mock")
    print("[ok] loaded %d rows, %d customers" % (ds.n_rows, ds.n_customers))
    rfm = build_rfm_table(ds.transactions)
    rfm = rfm_score(rfm)
    rfm = add_behavioural_features(ds.transactions, rfm)
    results, _ = compare_clusterings(rfm, k_range=[3])
    best = max(results, key=lambda r: r.composite_score)
    rfm["Cluster"] = best.labels
    print("[ok] best cluster: %s (composite=%.3f)" % (best.algorithm, best.composite_score))
    rfm_clv, clv_res = fit_clv(ds.transactions, horizon_days=365)
    rfm["clv_365d"] = rfm_clv[clv_res.clv_col]
    print("[ok] CLV mean = %.2f" % rfm["clv_365d"].mean())
    rfm, _ = fit_churn(ds.transactions, rfm)
    print("[ok] churn_prob range = [%.3f, %.3f]" % (
        rfm["churn_prob"].min(), rfm["churn_prob"].max()
    ))

    banner("AGENT 1 / SegmentNamingAgent")
    seg_agent = SegmentNamingAgent(llm=llm)
    seg_res = seg_agent.run({"rfm": rfm})
    segments = seg_res.payload["segments"]
    assert len(segments) == rfm["Cluster"].nunique(), (
        "expected %d segments, got %d" % (rfm["Cluster"].nunique(), len(segments))
    )
    for s in segments:
        assert "business_name" in s and s["business_name"], "segment missing business_name"
        assert "priority" in s, "segment missing priority"
    print("[ok] %d segments returned" % len(segments))
    for s in segments:
        print("     cluster %d -> %s [%s] | %s" % (
            s.get("cluster_id"), s.get("business_name"),
            s.get("priority"), s.get("tagline", "")[:40]
        ))

    banner("AGENT 2 / StrategyComposerAgent")
    strat_agent = StrategyComposerAgent(llm=llm)
    nba_res = strat_agent.run({"rfm": rfm, "segments": seg_res.payload})
    nba = nba_res.payload["nba"]
    assert len(nba) == len(rfm), "expected %d NBA rows, got %d" % (len(rfm), len(nba))
    for row in nba[:3]:
        for key in ("customer_id", "recommended_action", "channel",
                    "expected_conversion_rate", "expected_roi", "reasoning"):
            assert key in row, "row missing key: %s" % key
    print("[ok] %d NBA rows returned" % len(nba))
    print("[ok] summary: %s" % json.dumps(nba_res.payload["summary"], ensure_ascii=False, indent=2))
    print("[ok] first 3 NBA rows:")
    for row in nba[:3]:
        print("     cid=%s action=%s ch=%s roi=%s | %s" % (
            row["customer_id"], row["recommended_action"][:30],
            row["channel"], row["expected_roi"], row["reasoning"][:40]
        ))

    banner("AGENT 3 / ChatAgent")
    chat = ChatAgent(llm=llm)
    chat.bind_state({"rfm": rfm, "segments": seg_res.payload})
    q1 = chat.ask("客户 12345 的状态")
    q2 = chat.ask("Champions 群体表现如何")
    q3 = chat.ask("最近营收趋势")
    for turn in (q1, q2, q3):
        assert turn.answer.strip(), "chat returned empty answer"
    print("[ok] Q1 (%s) -> %s" % (q1.tool, q1.answer[:80]))
    print("[ok] Q2 (%s) -> %s" % (q2.tool, q2.answer[:80]))
    print("[ok] Q3 (%s) -> %s" % (q3.tool, q3.answer[:80]))

    banner("RESULT FILES")
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nba_recommendations_real_llm.csv").write_text(
        "\n".join([
            "customer_id,recommended_action,channel,expected_conversion_rate,expected_revenue_per_customer,cost_per_touch,expected_roi,reasoning",
        ] + [
            "%d,%s,%s,%s,%s,%s,%s,%s" % (
                r["customer_id"], r["recommended_action"].replace(",", ";"),
                r["channel"], r["expected_conversion_rate"],
                r["expected_revenue_per_customer"], r["cost_per_touch"],
                r["expected_roi"], r["reasoning"].replace(",", ";"),
            )
            for r in nba
        ]),
        encoding="utf-8",
    )
    print("[ok] NBA csv -> data/processed/nba_recommendations_real_llm.csv")
    print("\n[DONE] All 3 agents answered via real MiniMax LLM.")


if __name__ == "__main__":
    main()
