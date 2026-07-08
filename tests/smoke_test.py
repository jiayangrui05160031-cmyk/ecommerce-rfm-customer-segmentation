"""Smoke test: end-to-end pipeline validation in <30 seconds."""
from __future__ import annotations
import sys
import time
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_config, project_root  # noqa: E402
from src.data_sources import load_dataset  # noqa: E402
from src.features import build_rfm_table, rfm_score, add_behavioural_features  # noqa: E402
from src.models import compare_clusterings, fit_clv, fit_churn  # noqa: E402
from src.agents import SegmentNamingAgent, StrategyComposerAgent, ChatAgent  # noqa: E402
from src.reports import build_html_report  # noqa: E402


def test_smoke_pipeline():
    """End-to-end smoke. Must complete in <60 seconds."""
    cfg = get_config()
    assert cfg.llm.provider == "mock", "smoke test must use MockLLM"
    print("[smoke] using data source: mock")
    t0 = time.time()
    ds = load_dataset("mock")
    assert ds.n_rows > 100, "mock dataset too small: %d" % ds.n_rows
    print("[smoke] dataset loaded: %d rows, %d customers" % (ds.n_rows, ds.n_customers))

    rfm = build_rfm_table(ds.transactions)
    rfm = rfm_score(rfm)
    rfm = add_behavioural_features(ds.transactions, rfm)
    assert "avg_basket" in rfm.columns, "behavioural features missing"
    print("[smoke] features built: shape=%s" % (rfm.shape,))

    results, summary = compare_clusterings(rfm, k_range=[3, 4])
    assert len(results) > 0, "no clustering results"
    best = max(results, key=lambda r: r.composite_score)
    rfm["Cluster"] = best.labels
    print("[smoke] clustering best: %s (composite=%.3f)" % (best.algorithm, best.composite_score))

    rfm_clv, clv_res = fit_clv(ds.transactions, horizon_days=365)
    assert clv_res.clv_col in rfm_clv.columns, "CLV column missing"
    rfm["clv_365d"] = rfm_clv[clv_res.clv_col]
    print("[smoke] CLV ok: mean=%.2f" % rfm["clv_365d"].mean())

    rfm, churn_res = fit_churn(ds.transactions, rfm)
    assert "churn_prob" in rfm.columns, "churn_prob missing"
    print("[smoke] churn ok: auc=%s" % churn_res.auc)

    seg_agent = SegmentNamingAgent()
    seg_res = seg_agent.run({"rfm": rfm})
    segments = seg_res.payload["segments"]
    assert len(segments) == rfm["Cluster"].nunique(), "segments mismatch clusters"
    print("[smoke] segment naming ok: %d segments" % len(segments))
    for s in segments:
        assert "business_name" in s, "segment missing business_name"
        assert "priority" in s, "segment missing priority"

    strat_agent = StrategyComposerAgent()
    nba_res = strat_agent.run({"rfm": rfm, "segments": seg_res.payload})
    nba = nba_res.payload["nba"]
    assert len(nba) == len(rfm), "NBA row count mismatch"
    print("[smoke] strategy composer ok: %d NBA rows" % len(nba))

    chat = ChatAgent()
    chat.bind_state({"rfm": rfm, "segments": seg_res.payload})
    q1 = chat.ask("客户 12345 的状态")
    assert q1.tool == "query_customer"
    q2 = chat.ask("Champions 群体表现如何")
    assert q2.tool == "query_segment"
    q3 = chat.ask("最近营收趋势")
    assert q3.tool == "query_trend"
    print("[smoke] chat agent ok")

    out = build_html_report(
        rfm=rfm,
        segments_payload=seg_res.payload,
        clv_payload=clv_res.summary,
        nba_payload=nba_res.payload,
        chat_examples=[
            {"question": q1.question, "answer": q1.answer},
            {"question": q2.question, "answer": q2.answer},
            {"question": q3.question, "answer": q3.answer},
        ],
        dataset_name=ds.name,
        n_transactions=ds.n_rows,
        out_path=Path(tempfile.gettempdir()) / "ecommerce_rfm_smoke_report.html",
    )
    assert Path(out).exists(), "HTML report not generated"
    size = Path(out).stat().st_size
    assert size > 5000, "HTML report suspiciously small: %d bytes" % size
    print("[smoke] HTML report ok: %s (%d bytes)" % (out, size))

    duration = time.time() - t0
    assert duration < 60, "smoke test too slow: %.1fs" % duration
    print("\n[smoke] PASSED in %.2fs" % duration)


if __name__ == "__main__":
    test_smoke_pipeline()
