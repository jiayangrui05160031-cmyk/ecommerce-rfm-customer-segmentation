"""Base classes for all agents.

Each agent is a small wrapper around an LLM call. The contract:
    agent = SomeAgent(llm=...)
    result = agent.run(inputs: dict) -> AgentResult
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from src.agents.llm_factory import MockLLM, get_llm


@dataclass
class AgentResult:
    agent_name: str
    payload: dict
    raw_response: str = ""
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "agent_name": self.agent_name,
            "payload": self.payload,
            "raw_response": self.raw_response,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class BaseAgent:
    name: str = "base"

    def __init__(self, llm=None, prompt_loader=None):
        self.llm = llm or get_llm()
        self.prompt_loader = prompt_loader

    def _call_llm(self, prompt):
        resp = self.llm.invoke(prompt)
        content = getattr(resp, "content", str(resp))
        return content

    def _parse_json(self, text):
        s = text.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            s = "\n".join(line for line in lines if not line.strip().startswith("```"))
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            for opener, closer in (("{", "}"), ("[", "]")):
                start = s.find(opener)
                end = s.rfind(closer)
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(s[start:end + 1])
                    except json.JSONDecodeError:
                        pass
            raise

    def run(self, inputs):
        raise NotImplementedError
