"""Unified LLM factory.

Returns a LangChain-compatible chat model based on config. Supported:
  - mock    : deterministic canned responses (no API key needed)
  - deepseek: deepseek-chat via OpenAI-compatible API
  - openai  : gpt-4o-mini by default
  - anthropic: claude via anthropic-compatible API
  - ollama  : local Qwen / Llama via ollama server
  - minimax : MiniMax-Text-01 / abab series via MiniMax OpenAI-compatible API

When llm.provider == "mock" we always return MockLLM regardless of
other config, so CI / smoke tests never hit the network.
"""
from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass
import src.config as _config  # 用模块对象而非 import-as-name, 保证 monkey-patch 生效


class MockLLM:
    """Deterministic mock LLM for tests and offline dev."""

    def __init__(self, fixture_dir=None):
        self.calls = []
        self._fixture_dir = fixture_dir or ""

    def invoke(self, messages, **kwargs):
        if isinstance(messages, str):
            user_text = messages
        else:
            user_text = "\n".join(getattr(m, "content", str(m)) for m in messages)
        self.calls.append(user_text)
        agent_tag = "unknown"
        for tag in ("segment_naming", "strategy_composer", "chat_system"):
            if tag in user_text:
                agent_tag = tag
                break
        text = self._canned(agent_tag)
        return MockResponse(text)

    def _canned(self, agent_tag):
        if self._fixture_dir:
            f = os.path.join(self._fixture_dir, "%s.json" % agent_tag)
            if os.path.isfile(f):
                with open(f, "r", encoding="utf-8") as fh:
                    return fh.read()
        if agent_tag == "segment_naming":
            return json.dumps({
                "cluster_id": 0,
                "business_name": "高价值低频休眠客户",
                "english_name": "High-Value Hibernating",
                "tagline": "曾经的高消费用户，最近 90 天无购买",
                "priority": "P1",
                "core_pain": "复购唤醒",
                "core_desire": "回归高频消费",
            }, ensure_ascii=False)
        if agent_tag == "strategy_composer":
            actions = [
                "VIP 专属 9 折 + 私人客服回访",
                "会员日特权 + 积分翻倍",
                "首单 8 折复购券 + 欢迎邮件",
                "唤醒短信 + 大额折扣",
                "新品优先购 + 限量试装",
            ]
            rows = [
                {
                    "customer_id": i,
                    "recommended_action": random.choice(actions),
                    "channel": random.choice(["email", "sms", "app_push"]),
                    "expected_conversion_rate": round(random.uniform(0.05, 0.20), 3),
                    "expected_revenue_per_customer": round(random.uniform(50, 500), 2),
                    "cost_per_touch": round(random.uniform(0.5, 5.0), 2),
                    "expected_roi": round(random.uniform(2.0, 8.0), 2),
                    "reasoning": "基于 segment + churn + CLV 联合判定",
                }
                for i in range(5)
            ]
            return json.dumps(rows, ensure_ascii=False)
        return json.dumps({"text": "Mock response"}, ensure_ascii=False)

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


@dataclass
class MockResponse:
    content: str

    def __init__(self, content):
        self.content = content

    def __str__(self):
        return self.content


class MiniMaxChat:
    """Minimal OpenAI-compatible client for MiniMax (MiniMax).

    MiniMax exposes an OpenAI-style /v1/text/chatcompletion_v2 endpoint
    using the same request/response shape. We don't pull in the
    openai SDK so this remains a single-file dependency.
    """

    def __init__(self, api_key, model="MiniMax-Text-01",
                 base_url="https://api.minimaxi.com/v1",
                 temperature=0.3, max_tokens=1024, timeout=60):
        import urllib.request
        import urllib.error
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._request = urllib.request
        self._error = urllib.error

    def invoke(self, messages, **kwargs):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = [
                {"role": getattr(m, "type", "user"),
                 "content": getattr(m, "content", str(m))}
                for m in messages
            ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = self._request.Request(
            self.base_url + "/text/chatcompletion_v2",
            data=data, method="POST",
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with self._request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except self._error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError("MiniMax HTTP %s: %s" % (e.code, err_body))
        except self._error.URLError as e:
            raise RuntimeError("MiniMax URL error: %s" % e)
        parsed = json.loads(body)
        content = parsed["choices"][0]["message"]["content"]
        return MockResponse(content)

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


def get_llm(temperature=None, max_tokens=None):
    """Return an LLM client per current config. Falls back to MockLLM."""
    # 用 _config.get_config() 而不是 get_config(), 让 monkey-patch 能生效
    cfg = _config.get_config().llm
    provider = (cfg.provider or "mock").lower()
    temp = temperature if temperature is not None else cfg.temperature
    max_tok = max_tokens if max_tokens is not None else cfg.max_tokens
    if provider == "mock":
        return MockLLM()
    api_key = os.environ.get(cfg.api_key_env, "")
    if not api_key:
        return MockLLM()
    try:
        if provider in ("deepseek", "openai"):
            from langchain_openai import ChatOpenAI  # type: ignore
            kwargs = {
                "model": cfg.model,
                "temperature": temp,
                "max_tokens": max_tok,
                "api_key": api_key,
            }
            if cfg.base_url:
                kwargs["base_url"] = cfg.base_url
            elif provider == "deepseek":
                kwargs["base_url"] = "https://api.deepseek.com/v1"
            return ChatOpenAI(**kwargs)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic  # type: ignore
            return ChatAnthropic(
                model=cfg.model,
                temperature=temp, max_tokens=max_tok,
                api_key=api_key,
            )
        if provider == "ollama":
            from langchain_ollama import ChatOllama  # type: ignore
            return ChatOllama(
                model=cfg.model, base_url=cfg.ollama_base_url,
                temperature=temp,
            )
        if provider == "minimax":
            base_url = cfg.base_url or "https://api.minimaxi.com/v1"
            return MiniMaxChat(
                api_key=api_key, model=cfg.model,
                base_url=base_url, temperature=temp, max_tokens=max_tok,
            )
    except Exception:
        return MockLLM()
    return MockLLM()
