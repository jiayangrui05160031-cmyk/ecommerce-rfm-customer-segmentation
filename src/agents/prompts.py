"""Prompt templates for all agents.

Plain Python format strings so they work with the MockLLM and any
LangChain chat model.
"""
from __future__ import annotations


SEGMENT_NAMING_TEMPLATE = """你是一个电商 CRM 分析师。请根据以下客户分群画像数据，为每个 segment 起一个业务名称（中文优先）和一句话定位。

分群画像（每个 cluster 一行）:
{profile}

要求:
- business_name 用中文，长度不超过 10 个字
- english_name 是英文短名（如 High-Value Hibernating）
- tagline 一句话（不超过 30 字）
- priority 取 P0 / P1 / P2 / P3（P0 最重要）
- core_pain 是这个群体最大的痛点
- core_desire 是他们最想要的

请严格按 JSON 数组返回，每个元素对应一个 cluster，cluster_id 必须与输入一致。不要任何解释文字。
"""


STRATEGY_COMPOSER_TEMPLATE = """你是一个电商营销策略专家。下面是多个客户 segment 的画像，请你为每个 segment 生成一个营销策略模板（segment-level template）。

Segment 画像:
{segments}

输出格式要求:
- 严格按 JSON 对象返回
- 每个 key 是 segment_name（与输入完全一致）
- 每个 value 是一个对象，包含:
  - recommended_action: 简短中文动作（不超过 20 字）
  - channel: email / sms / app_push / in_app 之一（按 segment 特点选）
  - expected_conversion_rate: 0-1 之间的数字
  - expected_revenue_per_customer: 数字
  - cost_per_touch: 数字
  - expected_roi: 数字
  - reasoning: 一句话说明为什么这个策略适合这个 segment

不同 segment 应该有不同 channel，体现差异化。
不要任何解释文字，只返回 JSON 对象。
"""


CHAT_SYSTEM_TEMPLATE = """你是一个电商数据分析师助手。你可以使用以下工具查询数据:
- query_segment(segment_name, metric): 查询某个 segment 的画像
- query_customer(customer_id): 查询某个客户
- query_trend(metric, time_range): 查询趋势

请用中文回答，结合数据给业务洞察。不要编造数字。
"""


def render_segment_naming_prompt(profile_rows):
    return SEGMENT_NAMING_TEMPLATE.format(profile=_rows_to_table(profile_rows))


def render_strategy_composer_prompt(segment_rows):
    return STRATEGY_COMPOSER_TEMPLATE.format(segments=_rows_to_table(segment_rows, max_rows=20))


def render_chat_system_prompt():
    return CHAT_SYSTEM_TEMPLATE


def _rows_to_table(rows, max_rows=50):
    if not rows:
        return "(no data)"
    truncated = rows[:max_rows]
    keys = list(truncated[0].keys())
    lines = [str(keys)]
    for r in truncated:
        lines.append(str({k: r.get(k) for k in keys}))
    if len(rows) > max_rows:
        lines.append("... (" + str(len(rows) - max_rows) + " more rows)")
    return "\n".join(lines)
