"""AI Agent layer for narrative, segmentation and strategy generation.

Each agent is a small wrapper around an LLM call. The LLM acts as a
translator: it never invents numbers, only turns facts into prose.
"""
from .llm_factory import get_llm, MockLLM
from .base import BaseAgent, AgentResult
from .segment_namer import SegmentNamingAgent
from .strategy_composer import StrategyComposerAgent
from .chat_agent import ChatAgent

__all__ = [
    "get_llm", "MockLLM", "BaseAgent", "AgentResult",
    "SegmentNamingAgent", "StrategyComposerAgent", "ChatAgent",
]
