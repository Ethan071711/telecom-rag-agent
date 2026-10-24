"""Check cited sources and numerical facts before an answer reaches the user."""
from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal

from .textutil import extract_citations, is_refusal, strip_citations


QUANTITY_RE = re.compile(
    r"(?<![\d.])([+-]?\d+(?:\.\d+)?)\s*"
    r"(元|块|kbps|mbps|gb|mb|g|分钟|小时|个工作日|工作日|天|个月|月|周岁|岁|张|次|积分|个|%)",
    re.I,
)
UNITS = {'块': '元', 'g': 'gb', '周岁': '岁', '个工作日': '工作日', '个月': '月'}


def quantities(text: str) -> set[tuple[Decimal, str]]:
    text = unicodedata.normalize('NFKC', strip_citations(text)).lower()
    return {(Decimal(number), UNITS.get(unit, unit)) for number, unit in QUANTITY_RE.findall(text)}


def tool_facts(result: str) -> str:
    """Preserve units from structured tool keys, rather than accepting a bare number anywhere."""
    try:
        value = json.loads(result)
    except (ValueError, TypeError):
        return ''
    if not isinstance(value, dict) or 'error' in value:
        return ''
    facts = [result]

    def walk(item):
        if isinstance(item, dict):
            for key, val in item.items():
                if type(val) in (int, float):
                    unit = next((u for u in ('GB', '分钟', '元') if u in key), None)
                    if key in {'月费', '套外流量费', '套外通话费', '副卡费', '合计'}:
                        unit = '元'
                    if key == '副卡数':
                        unit = '张'
                    if unit:
                        facts.append(f'{val}{unit}')
                walk(val)
        elif isinstance(item, list):
            for val in item:
                walk(val)

    walk(value)
    return '\n'.join(facts)


def issues(answer: str, evidence: list[str], tools: list[dict]) -> list[str]:
    if is_refusal(answer) and not quantities(answer):
        return []
    cited = extract_citations(answer)
    errors = []
    if any(n < 1 or n > len(evidence) for n in cited):
        errors.append('引用了不存在的资料编号')
    supporting = [evidence[n-1] for n in cited if 1 <= n <= len(evidence)]
    supporting += [facts for call in tools if call['name'] != 'search_knowledge'
                   if (facts := tool_facts(call['result']))]
    if not supporting:
        errors.append('没有有效引用或成功的业务工具结果')
    supported = set().union(*(quantities(text) for text in supporting)) if supporting else set()
    missing = quantities(answer) - supported
    if missing:
        values = '、'.join(f'{number}{unit}' for number, unit in sorted(missing))
        errors.append('所引用资料及工具结果不支持这些数字和单位：'+values)
    # This gate checks references and numerical facts, not full semantic entailment.
    return errors


def fee_answer(tools: list[dict]) -> str | None:
    """Keep a computed bill and its charging conditions together, without model paraphrasing."""
    from .offline import OfflineLLM
    business = [call for call in tools if call['name'] != 'search_knowledge']
    if not business or any(call['name'] != 'calculate_fee' or not tool_facts(call['result'])
                           for call in business):
        return None
    return '\n'.join(OfflineLLM._summarize_tool(call['name'],call['result']) for call in business)


def extractive_fallback(answer: str, evidence: list[str], tools: list[dict],
                        source_numbers: list[int] | None = None) -> str:
    from .offline import OfflineLLM
    from .textutil import REFUSAL_TEXT

    for call in reversed(tools):
        if call['name'] in {'calculate_fee', 'recommend_plan', 'query_plans'} and tool_facts(call['result']):
            return OfflineLLM._summarize_tool(call['name'], call['result'])
    valid = [n for n in (source_numbers or extract_citations(answer)) if 1 <= n <= len(evidence)]
    if not valid:
        return REFUSAL_TEXT
    return '以下为已检索资料原文，请以这些条件为准：\n'+ '\n'.join(
        f'{evidence[n-1]} [{n}]' for n in valid[:8])
