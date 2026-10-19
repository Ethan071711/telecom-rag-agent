"""离线规则基线（不调用任何大模型）。

和真实大模型走完全相同的智能体循环，只是“决策”换成了规则：
- 工具路由：用正则识别套餐名、流量（GB）、通话分钟数、副卡数和“最省钱/以内”等意图；
- 回答生成：从排名前两位的检索片段里，挑出和问题词重合最多的两句话原样返回（抽取式），并标注引用编号；
- 拒答：重排分数最高的片段也低于阈值时，认为知识库里没有答案。

作用：没有 API Key 和 GPU 的环境也能一条命令跑通；同时作为对照组，衡量大模型带来的提升。
"""

from __future__ import annotations

import json
import re

from .llm import Reply, ToolCall
from .retriever import Retriever
from .textutil import REFUSAL_TEXT
from .tools import PlanBook

ANSWER_MIN_SCORE = 0.5  # 最相关片段的重排分数低于它就拒答（只用于离线基线）
CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4}
RECOMMEND_WORDS = ("最省钱", "最划算", "划算", "推荐", "选哪个", "哪个套餐", "哪款")
STUDENT_WORDS = ("学生", "大学", "校园", "研究生")


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")


def parse_usage(text: str) -> tuple[float | None, float | None, int | None]:
    """从一句话里解析 (流量GB, 通话分钟, 副卡数)。“5G套餐”里的 5G 是网络制式，先去掉。"""
    cleaned = re.sub(r"(?<![\d.])[45]G(?![Bb\d])", "", text)
    data = re.search(r"(\d+(?:\.\d+)?)\s*G(?:B)?", cleaned, re.I)
    minutes = re.search(r"(\d+)\s*分钟", cleaned)
    cards = re.search(r"(\d+|[一二两三四])\s*张副卡", cleaned)
    card_n = None
    if cards:
        card_n = int(cards.group(1)) if cards.group(1).isdigit() else CN_NUM[cards.group(1)]
    return (float(data.group(1)) if data else None,
            float(minutes.group(1)) if minutes else None,
            card_n)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；])|\n", text)
    sentences = []
    for part in parts:
        part = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", part).strip()
        if len(part) >= 6 and not set(part) <= set("|- "):
            sentences.append(part)
    return sentences


class OfflineLLM:
    def __init__(self, plans: PlanBook | None = None):
        self.plans = plans or PlanBook()

    def chat(self, messages: list[dict], tools: list[dict], context=None) -> Reply:
        last = messages[-1]
        if last["role"] == "tool":
            return Reply(content=self._summarize_tool(last["name"], last["content"]))
        user_texts = [m["content"] for m in messages if m["role"] == "user"]
        call = self._route(user_texts[-1], user_texts[:-1])
        if call:
            return Reply(tool_calls=[call])
        return Reply(content=self._extractive_answer(user_texts[-1], context or []))

    # ---- 工具路由 ----
    def _route(self, text: str, previous: list[str]) -> ToolCall | None:
        data, minutes, cards = parse_usage(text)
        if data is not None or minutes is not None:
            if any(w in text for w in RECOMMEND_WORDS):
                return ToolCall("call_0", "recommend_plan", {
                    "data_gb": data or 0, "voice_minutes": minutes or 0, "secondary_cards": cards or 0,
                    "is_student": any(w in text for w in STUDENT_WORDS)})
            plan = self.plans.resolve(text)
            for prev in reversed(previous):  # 多轮：本轮没说套餐名，就沿用上文提到的套餐
                if plan:
                    break
                plan = self.plans.resolve(prev)
            if plan:
                return ToolCall("call_0", "calculate_fee", {
                    "plan": plan["name"], "data_gb": data or 0, "voice_minutes": minutes or 0,
                    "secondary_cards": cards or 0})
        budget = re.search(r"(\d+)\s*元(?:以内|以下|之内)|不超过\s*(\d+)\s*元", text)
        if budget and "套餐" in text:
            return ToolCall("call_0", "query_plans", {
                "max_monthly_fee": float(budget.group(1) or budget.group(2)), "need_5g": "5G" in text.upper()})
        return None

    # ---- 把工具结果写成一句话 ----
    @staticmethod
    def _summarize_tool(name: str, content: str) -> str:
        data = json.loads(content)
        if "error" in data:
            return f"抱歉，{data['error']}。"
        if name == "calculate_fee":
            return f"按{data['套餐']}计算，本月费用合计 {_fmt(data['合计(元)'])} 元。计算过程：{data['计算过程']}。"
        if name == "recommend_plan":
            top = data["费用排名"][:3]
            others = "；".join(f"{r['套餐']} {_fmt(r['预计月费用(元)'])} 元" for r in top[1:])
            return (f"按您的用量（{data['用量']}），最省钱的是{data['最省钱的套餐']}，"
                    f"预计每月 {_fmt(top[0]['预计月费用(元)'])} 元；其次是：{others}。")
        if name == "query_plans":
            rows = data["套餐列表"]
            if not rows:
                return "没有找到符合条件的套餐。"
            return "符合条件的套餐有：" + "；".join(
                f"{r['套餐']}（月费{_fmt(r['月费(元)'])}元，{_fmt(r['国内通用流量(GB)'])}GB，"
                f"{_fmt(r['国内通话(分钟)'])}分钟{'，仅限学生' if r['仅限学生'] else ''}）" for r in rows) + "。"
        return content

    # ---- 抽取式回答 ----
    @staticmethod
    def _extractive_answer(question: str, hits: list) -> str:
        # 只有重排分数（0~1）能和阈值比较；关闭重排时 RRF 分数没有绝对含义，只要有检索结果就作答
        if not hits or (hits[0].reranked and hits[0].score < ANSWER_MIN_SCORE):
            return REFUSAL_TEXT
        q_tokens = set(Retriever.tokenize(question))
        scored = []
        for n, hit in enumerate(hits[:2], start=1):
            for i, sentence in enumerate(split_sentences(hit.chunk.text)):
                overlap = len(q_tokens & set(Retriever.tokenize(sentence)))
                scored.append((overlap, -n, -i, sentence, n))
        best = sorted(scored, reverse=True)[:2]
        best.sort(key=lambda x: (-x[1], -x[2]))  # 恢复原文顺序
        return "".join(f"{s.rstrip('。；')}[{n}]。" for _, _, _, s, n in best)
