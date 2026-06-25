"""Chat-with-Data Agent.

A lightweight ReAct-style agent. It detects intent via keyword
heuristics, calls the matching tool, then asks the LLM to format the
answer. Simpler than a full LangGraph setup, easy to test.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from src.agents.base import AgentResult, BaseAgent
from src.agents.prompts import render_chat_system_prompt
from src.agents.tools import TOOL_REGISTRY, ToolResult


@dataclass
class ChatTurn:
    question: str
    tool: str | None
    tool_input: dict
    tool_result: ToolResult | None
    answer: str


def _detect_intent(question):
    q = question.strip()
    m = re.search(r"(customer|客户)\s*(\d+)", q, flags=re.I)
    if m:
        return "query_customer", {"customer_id": int(m.group(2))}
    if any(k in q for k in ["趋势", "trend", "月份", "month", "增长"]):
        return "query_trend", {"metric": "revenue", "time_range": "all"}
    for seg in ["Champions", "Loyal", "New", "Hibernating", "At Risk"]:
        if seg.lower() in q.lower():
            return "query_segment", {"segment_name": seg, "metric": "all"}
    return "query_segment", {"segment_name": "all", "metric": "all"}


class ChatAgent(BaseAgent):
    name = "chat_system"

    def __init__(self, llm=None, state=None):
        super().__init__(llm=llm)
        self.state = state or {}

    def bind_state(self, state):
        self.state = state

    def ask(self, question):
        tool_name, tool_args = _detect_intent(question)
        tool_fn = TOOL_REGISTRY.get(tool_name)
        tool_result = None
        raw = ""
        if tool_fn is not None:
            try:
                tool_result = tool_fn(self.state, **tool_args)
                raw = tool_result.summary
            except Exception as exc:
                raw = "(tool error) " + str(exc)
        prompt = (
            render_chat_system_prompt()
            + "\n\n用户问题: " + question
            + "\n工具结果: " + raw
            + "\n请用 1-3 句话中文回答，要基于工具结果给出洞察。"
        )
        try:
            answer = self._call_llm(prompt)
        except Exception:
            answer = raw or "未能从数据中查询到相关信息。"
        return ChatTurn(
            question=question,
            tool=tool_name,
            tool_input=tool_args,
            tool_result=tool_result,
            answer=answer,
        )

    def run(self, inputs):
        question = inputs.get("question", "")
        turn = self.ask(question)
        return AgentResult(
            agent_name=self.name,
            payload={
                "question": turn.question,
                "tool": turn.tool,
                "tool_input": turn.tool_input,
                "answer": turn.answer,
            },
        )
