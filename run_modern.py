"""CLI entry: parse args, call run_pipeline(), print a short summary.

The full pipeline lives in src.pipeline.run_pipeline. This script is
intentionally thin — argparse + a single call. Anyone wanting to embed
the pipeline in a notebook / app should call src.analyze() instead.
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
from src.pipeline import run_pipeline, ALL_STEPS  # noqa: E402


def banner(text):
    print("\n" + "=" * 64)
    print("  " + text)
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--source", default=None,
                        help="Data source: retail_ii | olist | mock | donation. "
                             "Defaults to config.data.source.")
    parser.add_argument("--out", default=None, help="Output HTML report path.")
    parser.add_argument("--nba-out", default=None, help="NBA CSV output path.")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Steps to skip: " + " | ".join(ALL_STEPS))
    parser.add_argument("--profile", default=None,
                        help="Force a profile (retail|donation). Default: from Dataset.")
    args = parser.parse_args()

    cfg = get_config()
    source = args.source or cfg.data.source
    banner("E-COMMERCE RFM + AI AGENT PIPELINE (source=%s)" % source)
    t0 = time.time()

    ds = load_dataset(source)
    if args.profile:
        from src.data_sources import PROFILES
        ds.profile = PROFILES[args.profile]
    print("  规模: %d 行, %d 客户" % (ds.n_rows, ds.n_customers))
    print("  profile: %s" % ds.resolved_profile().name)

    # Resolve steps: --skip is the inverse of --steps
    skip = set(args.skip or [])
    steps = [s for s in ALL_STEPS if s not in skip]

    out_path = (
        Path(args.out) if args.out
        else project_root() / "reports" / "business_report.html"
    )
    nba_path = Path(args.nba_out) if args.nba_out else None

    res = run_pipeline(
        ds, steps=steps, cfg=cfg, out_dir=out_path, nba_out=nba_path,
    )

    if res.cluster is not None:
        print("  Cluster: %s (composite=%.3f)" %
              (res.cluster.algorithm, res.cluster.composite_score))
    if res.clv is not None:
        print("  CLV 概览: %s" % res.clv.summary)
    if res.churn is not None:
        print("  Churn AUC: %s" % res.churn.auc)
    if res.rules is not None:
        print("  关联规则: %d 条" % len(res.rules))
    if res.cohort is not None:
        print("  Cohort 矩阵: %s" % (res.cohort["retention_matrix"].shape,))
    if res.forecast is not None:
        print("  Forecast: %s, %d 月" %
              (res.forecast["method"], cfg.mining.forecast_horizon_months))
    if res.segments.get("segments"):
        for s in res.segments["segments"]:
            print("    cluster %d -> %s (%s)" % (
                s.get("cluster_id"), s.get("business_name"), s.get("priority"),
            ))
    if res.report_path:
        print("  报告: %s" % res.report_path)

    duration = time.time() - t0
    banner("ALL DONE in %.1fs" % duration)


if __name__ == "__main__":
    main()
