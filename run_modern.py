"""Modern end-to-end pipeline runner.

Runs the full analytics stack on a single dataset:
    load -> RFM features -> cluster -> CLV -> churn -> market basket
    -> cohort -> forecast -> AI agents -> HTML report.

The legacy `run_all.py` script is preserved for the original RFM+KMeans
flow. This runner adds the predictive + AI agent layers.

Usage:
    python run_modern.py --source mock     # quick demo
    python run_modern.py --source retail_ii
    python run_modern.py --source olist
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))

from src.config import get_config, project_root  # noqa: E402
from src.data_sources import load_dataset  # noqa: E402
from src.features import build_rfm_table, rfm_score, add_behavioural_features  # noqa: E402
from src.models import compare_clusterings, fit_clv, fit_churn  # noqa: E402
from src.mining import mine_association_rules, build_cohort_matrix, forecast_revenue  # noqa: E402
from src.agents import SegmentNamingAgent, StrategyComposerAgent, ChatAgent  # noqa: E402
from src.reports import build_html_report  # noqa: E402


def banner(text):
    print("\n" + "=" * 64)
    print("  " + text)
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None,
                        help="Data source: retail_ii | olist | mock. Defaults to config.")
    parser.add_argument("--out", default=None, help="Output HTML report path.")
    parser.add_argument("--nba-out", default=None, help="NBA CSV output path.")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Steps to skip: cluster / clv / churn / basket / cohort / forecast / agents")
    args = parser.parse_args()

    cfg = get_config()
    source = args.source or cfg.data.source
    banner("E-COMMERCE RFM + AI AGENT PIPELINE (source=%s)" % source)
    t0 = time.time()

    # 1. Load
    banner("STEP 1 - 数据加载")
    ds = load_dataset(source)
    print("  规模: %d 行, %d 客户" % (ds.n_rows, ds.n_customers))

    # 2. Features
    banner("STEP 2 - RFM + 行为特征")
    rfm = build_rfm_table(ds.transactions)
    rfm = rfm_score(rfm)
    rfm = add_behavioural_features(ds.transactions, rfm)
    print("  特征 shape: %s" % (rfm.shape,))

    # 3. Cluster
    if "cluster" not in args.skip:
        banner("STEP 3 - 多算法聚类对比")
        results, summary = compare_clusterings(
            rfm,
            k_range=cfg.clustering.k_range,
            algorithms=cfg.clustering.algorithms,
            random_state=cfg.clustering.random_state,
        )
        best = max(results, key=lambda r: r.composite_score)
        rfm["Cluster"] = best.labels
        print(summary.to_string(index=False))
        print("  选择: %s (composite=%.3f)" % (best.algorithm, best.composite_score))

    # 4. CLV
    if "clv" not in args.skip:
        banner("STEP 4 - CLV 建模 (BG/NBD + Gamma-Gamma)")
        rfm_clv, clv_res = fit_clv(
            ds.transactions,
            horizon_days=cfg.clv.prediction_horizon_days,
            penalizer=cfg.clv.bg_nbd_penalizer,
            time_unit_days=cfg.clv.time_unit_days,
        )
        rfm["clv_365d"] = rfm_clv[clv_res.clv_col]
        print("  CLV 概览: %s" % clv_res.summary)

    # 5. Churn
    if "churn" not in args.skip:
        banner("STEP 5 - Churn 预测 (LightGBM)")
        rfm, churn_res = fit_churn(
            ds.transactions, rfm,
            definition_days=cfg.churn.definition_days,
            random_state=cfg.churn.random_state,
        )
        print("  AUC: %s" % churn_res.auc)
        if churn_res.importance:
            top = sorted(churn_res.importance.items(), key=lambda x: -x[1])[:5]
            print("  Top features: " + ", ".join("%s=%.0f" % (k, v) for k, v in top))

    # 6. Market basket
    rules = None
    if "basket" not in args.skip:
        banner("STEP 6 - 关联规则挖掘")
        rules = mine_association_rules(
            ds.transactions,
            min_support=cfg.mining.market_basket_min_support,
        )
        print("  规则数: %d" % len(rules))
        if len(rules) > 0 and "lift" in rules.columns:
            print("  Top 3 lift: " + ", ".join(
                "%.2f" % v for v in rules["lift"].head(3) if pd_not_nan(v)
            ))

    # 7. Cohort
    if "cohort" not in args.skip:
        banner("STEP 7 - Cohort 留存")
        cohort = build_cohort_matrix(ds.transactions, period=cfg.mining.cohort_period)
        print("  留存矩阵 shape: %s" % (cohort["retention_matrix"].shape,))

    # 8. Forecast
    fcst = None
    if "forecast" not in args.skip:
        banner("STEP 8 - 营收预测")
        fcst = forecast_revenue(ds.transactions, horizon_months=cfg.mining.forecast_horizon_months)
        print("  方法: %s, 预测期: %d 月" % (fcst["method"], cfg.mining.forecast_horizon_months))

    # 9. Agents
    seg_payload = {}
    nba_payload = {"nba": []}
    chat_examples = []
    if "agents" not in args.skip:
        banner("STEP 9 - AI Agents")
        seg_agent = SegmentNamingAgent()
        seg_res = seg_agent.run({"rfm": rfm})
        seg_payload = seg_res.payload
        print("  Segment Naming: %d segments" % len(seg_payload["segments"]))
        for s in seg_payload["segments"]:
            print("    cluster %d -> %s (%s)" % (
                s.get("cluster_id"), s.get("business_name"), s.get("priority")
            ))
        strat_agent = StrategyComposerAgent()
        nba_res = strat_agent.run({"rfm": rfm, "segments": seg_payload})
        nba_payload = nba_res.payload
        if args.nba_out:
            Path(args.nba_out).parent.mkdir(parents=True, exist_ok=True)
            pd_save_csv(nba_payload["nba"], args.nba_out)
            print("  NBA CSV -> %s" % args.nba_out)
        chat = ChatAgent()
        chat.bind_state({"rfm": rfm, "segments": seg_payload, "history": fcst["history"] if fcst else None})
        for q in ["客户 12345 的状态", "Champions 群体表现如何", "最近营收趋势"]:
            turn = chat.ask(q)
            chat_examples.append({"question": q, "answer": turn.answer})
        print("  Chat: 3 example queries answered")

    # 10. HTML report
    banner("STEP 10 - HTML 业务报告")
    out_path = Path(args.out) if args.out else project_root() / "reports" / "business_report.html"
    out = build_html_report(
        rfm=rfm,
        segments_payload=seg_payload,
        clv_payload=(clv_res.summary if "clv" not in args.skip else None),
        nba_payload=nba_payload,
        chat_examples=chat_examples,
        dataset_name=ds.name,
        n_transactions=ds.n_rows,
        out_path=out_path,
    )
    print("  报告: %s" % out)

    duration = time.time() - t0
    banner("ALL DONE in %.1fs" % duration)


def pd_not_nan(v):
    try:
        return v == v  # NaN != NaN
    except Exception:
        return False


def pd_save_csv(rows, path):
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
