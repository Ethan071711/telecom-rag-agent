"""客服智能体：检索前置 + 模型自主决定是否调用工具 + 多轮对话记忆 + 引用回溯。

一轮对话的流程：
1. 拼检索 query：追问（“那毕业以后呢？”）太短或带指代词时，拼上上一轮的问题，补全上下文；
2. 混合检索 + 重排，得到带编号的参考资料，写进系统提示；
3. 调用大模型。模型可以返回工具调用（查套餐、算费用、推荐套餐、再检索），程序执行后把结果交回模型，
   最多循环 MAX_TOOL_STEPS 次，直到模型给出最终回答；
4. 从回答里解析 [n] 引用编号，对应回具体的文档片段，一起返回给用户。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Settings
from .prompts import NO_CONTEXT, SYSTEM_PROMPT
from .retriever import Hit, Retriever
from .textutil import extract_citations, is_refusal, strip_citations
from .tools import TOOL_SPECS, PlanBook

FOLLOWUP_HINTS = ("那", "它", "这个", "那个", "那边", "这边", "上面", "刚才", "以后", "呢", "还有", "该")


class EvidencePool:
    """本轮对话里给模型看过的全部资料片段，编号从 1 开始，也就是引用编号。"""

    def __init__(self):
        self.hits: list[Hit] = []

    def add(self, hits: list[Hit]) -> list[int]:
        numbers = []
        for hit in hits:
            existing = next((i for i, h in enumerate(self.hits) if h.chunk.chunk_id == hit.chunk.chunk_id), None)
            if existing is None:
                self.hits.append(hit)
                existing = len(self.hits) - 1
            numbers.append(existing + 1)
        return numbers

    def render(self, numbers: list[int] | None = None) -> str:
        numbers = numbers or list(range(1, len(self.hits) + 1))
        blocks = []
        for n in numbers:
            c = self.hits[n - 1].chunk
            blocks.append(f"[{n}]《{c.title}》{c.section}\n{c.text}")
        return "\n\n".join(blocks)

    def get(self, n: int) -> Hit | None:
        return self.hits[n - 1] if 1 <= n <= len(self.hits) else None


@dataclass
class AgentResult:
    question: str
    answer: str
    retrieval_query: str
    evidence: list[Hit]                     # 给模型看过的全部片段，下标+1 即引用编号
    cited: list[int]                        # 回答中实际引用（且有效）的编号
    tool_calls: list[dict] = field(default_factory=list)
    refused: bool = False
    latency_s: float = 0.0
    llm_calls: int = 0                      # 本轮调用大模型的次数（1 = 直接回答，每多一轮工具调用 +1）

    @property
    def cited_hits(self) -> list[Hit]:
        return [self.evidence[n - 1] for n in self.cited]

    def format(self, snippet_len: int = 60) -> str:
        lines = [self.answer]
        if self.tool_calls:
            lines.append("\n调用的工具：" + "；".join(
                f"{t['name']}({', '.join(f'{k}={v}' for k, v in t['arguments'].items())})" for t in self.tool_calls))
        if self.cited:
            lines.append("\n引用片段：")
            for n in self.cited:
                c = self.evidence[n - 1].chunk
                text = c.text.replace("\n", " ")
                lines.append(f"  [{n}]《{c.title}》{c.section}：{text[:snippet_len]}{'…' if len(text) > snippet_len else ''}")
        return "\n".join(lines)


class TelecomAgent:
    def __init__(self, retriever: Retriever, llm, settings: Settings, plans: PlanBook | None = None):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings
        self.plans = plans or PlanBook()
        self.history: list[dict] = []  # 只保存用户问题和最终回答的纯文本

    def reset(self) -> None:
        self.history.clear()

    def retrieval_query(self, question: str) -> str:
        previous = [m["content"] for m in self.history if m["role"] == "user"]
        if previous and (len(question) <= 15 or any(h in question for h in FOLLOWUP_HINTS)):
            return f"{previous[-1]} {question}"
        return question

    def ask(self, question: str) -> AgentResult:
        t0 = time.time()
        query = self.retrieval_query(question)
        pool = EvidencePool()
        pool.add(self.retriever.search(query))

        system = SYSTEM_PROMPT.format(plan_names="、".join(self.plans.names), context=pool.render() or NO_CONTEXT)
        messages = [{"role": "system", "content": system}]
        messages += self.history[-2 * self.settings.history_turns:]
        messages.append({"role": "user", "content": question})

        tool_log: list[dict] = []
        answer = None
        llm_calls = 0
        for _ in range(self.settings.max_tool_steps + 1):
            reply = self.llm.chat(messages, TOOL_SPECS, context=pool.hits)
            llm_calls += 1
            if not reply.tool_calls:
                answer = reply.content.strip()
                break
            messages.append({"role": "assistant", "content": reply.content, "tool_calls": reply.tool_calls,
                             "raw": reply.raw})
            for call in reply.tool_calls:
                result = self._run_tool(call.name, call.arguments, pool)
                tool_log.append({"name": call.name, "arguments": call.arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})
        if not answer:
            answer = "抱歉，这个问题我暂时无法完成，请换个问法或联系人工客服。"

        cited = [n for n in extract_citations(answer) if pool.get(n)]
        # 历史里去掉引用编号：下一轮的资料编号会重新分配，旧编号会误导模型
        self.history += [{"role": "user", "content": question},
                         {"role": "assistant", "content": strip_citations(answer)}]
        return AgentResult(question, answer, query, list(pool.hits), cited, tool_log,
                           refused=is_refusal(answer), latency_s=time.time() - t0, llm_calls=llm_calls)

    def _run_tool(self, name: str, arguments: dict, pool: EvidencePool) -> str:
        if name == "search_knowledge":
            hits = self.retriever.search(str(arguments.get("query", "")))
            return pool.render(pool.add(hits)) if hits else "没有检索到相关资料。"
        return self.plans.run(name, arguments)
