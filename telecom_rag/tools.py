"""智能体可以调用的工具。

- query_plans：按条件查询套餐（结构化数据，来自 data/plans.json）
- calculate_fee：套餐月费用计算器（套外流量按 GB 向上取整、150 元封顶、套外通话、副卡）
- recommend_plan：按用量算出所有套餐的费用并排序，回答“哪个最省钱”
- search_knowledge：再检索一次知识库（实现在 agent.py，因为需要检索器和引用编号）

大模型做多步算术容易出错，所以费用相关的问题一律交给确定性的 Python 函数计算，模型只负责理解问题和组织答案。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .config import PLANS_PATH

TOOL_SPECS: list[dict] = [
    {
        "name": "search_knowledge",
        "description": "在青禾通信业务知识库中检索资料。系统提示里的参考资料不够回答时，换一个更具体的关键词再检索。"
                       "返回带编号的资料片段，回答时用编号引用。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索关键词或完整问题"}},
            "required": ["query"],
        },
    },
    {
        "name": "query_plans",
        "description": "按条件查询在售套餐的月费、流量、通话、网络制式和副卡规则。"
                       "用户问“有哪些套餐”“某个价位以内的套餐”“流量多的套餐”时调用。所有条件都可以不填。",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "套餐名称关键词，如“59元”“校园卡”"},
                "max_monthly_fee": {"type": "number", "description": "月费上限（元）"},
                "min_data_gb": {"type": "number", "description": "国内通用流量下限（GB）"},
                "need_5g": {"type": "boolean", "description": "是否只要 5G 套餐"},
            },
            "required": [],
        },
    },
    {
        "name": "calculate_fee",
        "description": "计算某个套餐在给定用量下的当月总费用，包括套外流量费（按GB向上取整，每月封顶150元）、"
                       "套外通话费和收费副卡的功能费。用户给出套餐和用量、问要交多少钱时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {"type": "string", "description": "套餐名称，如“5G畅享59元”"},
                "data_gb": {"type": "number", "description": "当月国内通用流量用量（GB）；有副卡时填全家合计"},
                "voice_minutes": {"type": "number", "description": "当月国内主叫通话分钟数，没提到就填 0"},
                "secondary_cards": {"type": "integer", "description": "副卡数量，没有就填 0"},
            },
            "required": ["plan", "data_gb"],
        },
    },
    {
        "name": "recommend_plan",
        "description": "根据每月流量和通话用量，计算各套餐的月费用并从低到高排序，用于回答“哪个套餐最省钱/最划算”。",
        "parameters": {
            "type": "object",
            "properties": {
                "data_gb": {"type": "number", "description": "每月国内通用流量用量（GB）"},
                "voice_minutes": {"type": "number", "description": "每月国内主叫通话分钟数，没提到就填 0"},
                "is_student": {"type": "boolean", "description": "是否为在校学生（学生可以办青春校园卡）"},
                "secondary_cards": {"type": "integer", "description": "需要的副卡数量，默认 0"},
            },
            "required": ["data_gb"],
        },
    },
]


def _num(value, name: str, units: str = "", integer: bool = False) -> float:
    """Accept a complete number with an optional matching unit; never guess missing usage."""
    if type(value) not in (int, float, str):
        raise ValueError(f"{name}必须是明确的非负数字")
    if isinstance(value, str):
        suffix = rf"(?:{units})?" if units else ""
        match = re.fullmatch(rf"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*{suffix}\s*", value, re.I)
        if not match:
            raise ValueError(f"{name}无法识别，请提供明确的数字和正确单位")
        value = match.group(1)
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1_000_000_000:
        raise ValueError(f"{name}必须是0到1000000000之间的有限数字")
    if integer and not number.is_integer():
        raise ValueError(f"{name}必须是整数，不能自动取整")
    return number


def _boolean(value, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name}必须是true或false，请确认后重试")
    return value


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")


def _clean(x: float) -> int | float:
    """189.0 -> 189，126.5 保持不变，让工具返回的 JSON 更好读。"""
    return int(x) if float(x).is_integer() else round(x, 2)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower().replace("套餐", "")


class PlanBook:
    def __init__(self, path: Path = PLANS_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.plans: list[dict] = data["plans"]
        self.rules: dict = data["rules"]

    @property
    def names(self) -> list[str]:
        return [p["name"] for p in self.plans]

    def resolve(self, text: str) -> dict | None:
        """把“59元套餐”“畅享59”“校园卡”等说法对应到具体套餐；匹配不到返回 None。"""
        if not text:
            return None
        t = _normalize(text)
        best: tuple[int, dict] | None = None
        for plan in self.plans:
            for alias in [plan["name"], plan["id"], *plan["aliases"]]:
                a = _normalize(alias)
                # 以数字开头的别名要求前面不是数字，避免“129元”误匹配“29元”
                pattern = (r"(?<![\d.])" if a[:1].isdigit() else "") + re.escape(a)
                if a and re.search(pattern, t) and (best is None or len(a) > best[0]):
                    best = (len(a), plan)
        if best:
            return best[1]
        for fee in re.findall(r"(?<![\d.])(\d+)\s*元", text):
            matches = [p for p in self.plans if p["monthly_fee"] == int(fee)]
            if len(matches) == 1:
                return matches[0]
        if re.fullmatch(r"\d+", t):
            matches = [p for p in self.plans if p["monthly_fee"] == int(t)]
            if len(matches) == 1:
                return matches[0]
        return None

    # ---------------- 工具实现 ----------------
    def query_plans(self, keyword: str | None = None, max_monthly_fee=None, min_data_gb=None,
                    need_5g: bool | None = None) -> dict:
        if keyword is not None and not isinstance(keyword, str):
            raise ValueError("套餐关键词必须是文字")
        fee_limit = _num(max_monthly_fee, "月费上限", "元") if max_monthly_fee is not None else None
        data_min = _num(min_data_gb, "流量下限", "GB|G") if min_data_gb is not None else None
        if need_5g is not None:
            need_5g = _boolean(need_5g, "是否需要5G")
        rows = []
        target = self.resolve(keyword) if keyword else None
        for p in self.plans:
            if keyword:
                if target is not None and target is not p:
                    continue
                if target is None and _normalize(keyword) not in _normalize(p["name"] + "".join(p["aliases"])):
                    continue
            if fee_limit is not None and p["monthly_fee"] > fee_limit:
                continue
            if data_min is not None and p["data_gb"] < data_min:
                continue
            if need_5g and p["network"] != "5G":
                continue
            rows.append({
                "套餐": p["name"], "月费(元)": p["monthly_fee"], "国内通用流量(GB)": p["data_gb"],
                "国内通话(分钟)": p["voice_minutes"], "网络": p["network"],
                "套外流量(元/GB)": p["extra_data_yuan_per_gb"], "套外通话(元/分钟)": p["extra_voice_yuan_per_min"],
                "副卡": f"免费{p['free_secondary_cards']}张，最多{p['max_secondary_cards']}张",
                "仅限学生": p["student_only"], "说明": "；".join(p["benefits"]),
            })
        return {"符合条件的套餐数": len(rows), "套餐列表": rows}

    def calculate_fee(self, plan: str, data_gb, voice_minutes=0, secondary_cards=0) -> dict:
        if not isinstance(plan, str):
            raise ValueError("套餐名称必须是文字")
        p = self.resolve(plan)
        if p is None:
            return {"error": f"没有找到套餐“{plan}”，可选套餐：{'、'.join(self.names)}"}
        data = _num(data_gb, "流量用量", "GB|G")
        voice = _num(voice_minutes, "通话分钟数", "分钟")
        cards = int(_num(secondary_cards, "副卡数量", "张(?:副卡)?", integer=True))
        if cards > p["max_secondary_cards"]:
            return {"error": f"{p['name']}最多只能添加 {p['max_secondary_cards']} 张副卡"}

        extra_gb = max(0, math.ceil(round(data - p["data_gb"], 6)))  # 不足 1GB 按 1GB 计
        raw_data_fee = extra_gb * p["extra_data_yuan_per_gb"]
        cap = self.rules["extra_data_monthly_cap_yuan"]
        data_fee = min(raw_data_fee, cap)
        extra_min = max(0, math.ceil(round(voice - p["voice_minutes"], 6)))
        voice_fee = round(extra_min * p["extra_voice_yuan_per_min"], 2)
        paid_cards = max(0, cards - p["free_secondary_cards"])
        card_fee = paid_cards * p["secondary_card_fee"]
        total = round(p["monthly_fee"] + data_fee + voice_fee + card_fee, 2)

        data_note = f"{extra_gb}GB×{_fmt(p['extra_data_yuan_per_gb'])}元={_fmt(raw_data_fee)}元"
        if raw_data_fee > cap:
            data_note += f"，超过每月{_fmt(cap)}元封顶，按{_fmt(cap)}元收取（当月剩余时间降速至{self.rules['speed_after_cap']}）"
        steps = (f"月费{_fmt(p['monthly_fee'])} + 套外流量{_fmt(data_fee)}（{data_note}）"
                 f" + 套外通话{_fmt(voice_fee)}（{extra_min}分钟×{_fmt(p['extra_voice_yuan_per_min'])}元）"
                 f" + 副卡{_fmt(card_fee)}（收费副卡{paid_cards}张） = {_fmt(total)}元")
        return {
            "套餐": p["name"], "月费": p["monthly_fee"],
            "套餐内流量(GB)": p["data_gb"], "实际流量(GB)": _clean(data), "套外流量(GB,向上取整)": extra_gb,
            "套外流量费": _clean(data_fee), "触发封顶": raw_data_fee > cap,
            "套餐内通话(分钟)": p["voice_minutes"], "实际通话(分钟)": _clean(voice), "套外通话(分钟)": extra_min,
            "套外通话费": _clean(voice_fee), "副卡数": cards, "副卡费": _clean(card_fee),
            "合计(元)": _clean(total), "计算过程": steps,
        }

    def recommend_plan(self, data_gb, voice_minutes=0, is_student: bool = False, secondary_cards=0) -> dict:
        ranking = []
        data = _num(data_gb, "流量用量", "GB|G")
        voice = _num(voice_minutes, "通话分钟数", "分钟")
        cards = int(_num(secondary_cards, "副卡数量", "张(?:副卡)?", integer=True))
        is_student = _boolean(is_student, "学生身份")
        for p in self.plans:
            if p["student_only"] and not is_student:
                continue
            if p["id"] == "family_199":  # 含宽带的融合套餐不和个人套餐直接比价
                continue
            if cards > p["max_secondary_cards"]:
                continue
            fee = self.calculate_fee(p["name"], data, voice, cards)
            ranking.append({"套餐": p["name"], "预计月费用(元)": fee["合计(元)"], "网络": p["network"],
                            "计算过程": fee["计算过程"]})
        ranking.sort(key=lambda r: r["预计月费用(元)"])
        if not ranking:
            return {"error": "没有支持该副卡数量的可推荐套餐，请确认数量或联系人工客服"}
        return {
            "用量": f"流量{_fmt(data)}GB，通话{_fmt(voice)}分钟，副卡{cards}张",
            "最省钱的套餐": ranking[0]["套餐"] if ranking else None,
            "费用排名": ranking,
            "说明": "不含家庭融合199元（含宽带，需要宽带时再单独比较）；"
                    + ("已包含仅限学生办理的青春校园卡，校园定向流量不计入。" if is_student else "非学生不能办理青春校园卡。"),
        }

    def run(self, name: str, arguments: dict) -> str:
        """供智能体调用：执行工具并返回 JSON 字符串。"""
        funcs = {"query_plans": self.query_plans, "calculate_fee": self.calculate_fee,
                 "recommend_plan": self.recommend_plan}
        if name not in funcs:
            return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)
        try:
            if not isinstance(arguments, dict):
                raise TypeError("工具参数必须是对象")
            result = funcs[name](**arguments)
        except (TypeError, ValueError, OverflowError) as exc:
            result = {"error": f"参数错误：{exc}；请确认参数后重试"}
        return json.dumps(result, ensure_ascii=False, allow_nan=False)
