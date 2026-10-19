"""智能体和评测共用的文本工具：拒答识别、引用编号解析、数字事实抽取。"""

from __future__ import annotations

import re
import unicodedata

REFUSAL_TEXT = "抱歉，知识库中没有找到相关信息，建议通过青禾通信 App 在线客服或到营业厅咨询人工客服。"

# 系统提示要求模型拒答时用固定开头；这里再兼容几种常见说法。只看回答开头，避免误伤正常回答里的“没有提到某细节”。
REFUSAL_PATTERNS = (
    "没有找到", "未找到", "没有查到", "没有相关信息", "暂无相关", "无相关信息", "没有提到", "没有提及",
    "未提及", "未提到", "不在知识库", "知识库中没有", "知识库里没有", "资料中没有", "资料里没有", "无法回答",
)
REFUSAL_WINDOW = 80

CITE_RE = re.compile(r"[\[【]\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*[\]】]")
NUM_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(元|块|kbps|mbps|gb|g|mb|分钟|小时|个工作日|天|个月|周岁|岁|张|次|积分|个|%)"
)


def normalize(text: str) -> str:
    """全角转半角、转小写、去空白，并把 GB 统一写成 G，方便做包含判断。"""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", "", text).replace("gb", "g")


def is_refusal(answer: str) -> bool:
    head = normalize(answer)[:REFUSAL_WINDOW]
    return any(p in head for p in REFUSAL_PATTERNS)


def extract_citations(answer: str) -> list[int]:
    """解析 [1]、[1][3]、[1,2]、【2】 这类引用编号，按出现顺序去重。"""
    numbers: list[int] = []
    for group in CITE_RE.findall(answer or ""):
        for n in re.split(r"[,，、]", group):
            n = int(n.strip())
            if n not in numbers:
                numbers.append(n)
    return numbers


def strip_citations(answer: str) -> str:
    return CITE_RE.sub("", answer or "")


def numeric_claims(answer: str) -> list[str]:
    """抽取回答里带单位的数字（如 189元、120天、1Mbps），返回数字部分。"""
    return [num for num, _ in NUM_UNIT_RE.findall(normalize(strip_citations(answer)))]
