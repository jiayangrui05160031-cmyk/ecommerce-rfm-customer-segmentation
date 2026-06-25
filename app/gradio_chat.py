"""Gradio Chat-with-Data demo.

Run:
    python -m app.gradio_chat

Then open http://localhost:7860 in your browser.
The app first runs the full pipeline (load -> RFM -> cluster -> CLV ->
churn -> agents), then exposes a chat interface where you can ask
business questions answered by the ChatAgent.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import get_config, project_root  # noqa: E402
from src.data_sources import load_dataset  # noqa: E402
from src.features import build_rfm_table, rfm_score, add_behavioural_features  # noqa: E402
from src.models import compare_clusterings, fit_clv, fit_churn  # noqa: E402
from src.agents import SegmentNamingAgent, StrategyComposerAgent, ChatAgent  # noqa: E402


EXAMPLE_QUESTIONS = [
    "客户 12345 的状态",
    "客户 67890 的状态",
    "Champions 群体表现如何",
    "Hibernating 群体表现如何",
    "最近营收趋势",
    "Loyal 群体表现如何",
]


def build_state(source="mock", n_customers=200):
    """Run the full pipeline and return state dict for the chat agent."""
    cfg = get_config()
    print("[gradio] loading data source: %s" % source)
    ds = load_dataset(source)
    rfm = build_rfm_table(ds.transactions)
    rfm = rfm_score(rfm)
    rfm = add_behavioural_features(ds.transactions, rfm)
    results, _ = compare_clusterings(rfm, k_range=[3, 4])
    best = max(results, key=lambda r: r.composite_score)
    rfm["Cluster"] = best.labels
    rfm_clv, _ = fit_clv(ds.transactions, horizon_days=cfg.clv.prediction_horizon_days)
    rfm["clv_365d"] = rfm_clv[clv_clv_col(rfm_clv, cfg)]
    rfm, _ = fit_churn(ds.transactions, rfm)
    seg_agent = SegmentNamingAgent()
    seg_res = seg_agent.run({"rfm": rfm})
    return {
        "rfm": rfm,
        "segments": seg_res.payload,
        "history": None,
    }


def clv_clv_col(rfm_clv, cfg):
    return "clv_%dd" % cfg.clv.prediction_horizon_days


def build_interface(state, chat):
    import gradio as gr

    def respond(question, history):
        if not question.strip():
            return ""
        turn = chat.ask(question)
        return turn.answer

    return gr.ChatInterface(
        respond,
        examples=EXAMPLE_QUESTIONS,
        title="电商 RFM Chat-with-Data",
        description="基于 RFM + CLV + Churn + Next-Best-Action 的智能问答",
        theme="default",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="mock")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--smoke-check", action="store_true",
                        help="Verify interface can build without launching.")
    args = parser.parse_args()

    state = build_state(source=args.source)
    chat = ChatAgent()
    chat.bind_state(state)
    if args.smoke_check:
        print("[gradio] smoke check OK - state built, chat agent bound")
        return
    iface = build_interface(state, chat)
    print("[gradio] launching on port %d ..." % args.port)
    iface.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
