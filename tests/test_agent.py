"""用假的检索器和脚本化的假大模型测试智能体循环，不需要下载任何模型。"""

from telecom_rag.agent import TelecomAgent
from telecom_rag.config import Settings
from telecom_rag.documents import Chunk
from telecom_rag.evaluate import grade
from telecom_rag.llm import Reply, ToolCall
from telecom_rag.offline import OfflineLLM, parse_usage
from telecom_rag.retriever import Hit

CHUNKS = [
    Chunk("09_suspension#0", "09_suspension", "停机保号、欠费停机与复机", "停机保号", "保号费 5 元/月，单次最长可以办理 12 个月。"),
    Chunk("01_plans_overview#5", "01_plans_overview", "个人及家庭套餐资费总览", "5G畅享59元", "月费 59 元，含 60GB 国内通用流量和 300 分钟国内通话。"),
]


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k=None, mode="hybrid_rerank", apply_threshold=True):
        self.queries.append(query)
        return [Hit(c, 0.9) for c in CHUNKS]


class ScriptedLLM:
    """第一次调用返回工具调用，第二次根据工具结果作答。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools, context=None):
        self.seen.append(messages)
        return self.replies.pop(0)


def settings():
    s = Settings.from_env()
    s.max_tool_steps, s.history_turns = 4, 6
    return s


def test_tool_loop_and_citations():
    llm = ScriptedLLM([
        Reply(tool_calls=[ToolCall("c1", "calculate_fee", {"plan": "5G畅享59元", "data_gb": 75, "voice_minutes": 300})]),
        Reply(content="本月合计 104 元 [2]。"),
    ])
    agent = TelecomAgent(FakeRetriever(), llm, settings())
    result = agent.ask("59元套餐用了75GB流量、300分钟要交多少钱？")
    assert result.answer == "本月合计 104 元 [2]。"
    assert result.tool_calls[0]["name"] == "calculate_fee"
    assert '"合计(元)": 104' in result.tool_calls[0]["result"]
    assert [h.chunk.doc_id for h in result.cited_hits] == ["01_plans_overview"]
    # 第二次调用时，模型能看到 assistant 的工具调用和工具结果
    roles = [m["role"] for m in llm.seen[1]]
    assert roles[-2:] == ["assistant", "tool"]
    # 历史里的回答去掉了引用编号
    assert agent.history[-1]["content"] == "本月合计 104 元 。"


def test_followup_question_uses_previous_turn_for_retrieval():
    retriever = FakeRetriever()
    llm = ScriptedLLM([Reply(content="可以办理停机保号 [1]。"), Reply(content="最长 12 个月 [1]。")])
    agent = TelecomAgent(retriever, llm, settings())
    agent.ask("我想办理停机保号。")
    agent.ask("最长能保多久？")
    assert retriever.queries[-1] == "我想办理停机保号。 最长能保多久？"


def test_grade_flags_unsupported_numbers():
    llm = ScriptedLLM([Reply(content="保号费 8 元/月，最长 12 个月 [1]。")])
    agent = TelecomAgent(FakeRetriever(), llm, settings())
    result = agent.ask("停机保号多少钱？")
    item = {"id": "x", "type": "kb", "question": "停机保号多少钱？", "must_include": [["5元"]],
            "gold_docs": ["09_suspension"]}
    d = grade(item, result)
    assert d["citation_hit"] is True
    assert d["correct"] is False
    assert d["unsupported"] and "8" in d["unsupported_reason"]


def test_offline_usage_parser():
    assert parse_usage("5G畅享39元套餐用了70GB流量、打了100分钟电话") == (70.0, 100.0, None)
    assert parse_usage("加了2张副卡，用了120.5GB") == (120.5, None, 2)
    assert parse_usage("有两张副卡") == (None, None, 2)


def test_offline_refusal_only_uses_rerank_scores():
    llm = OfflineLLM()
    messages = [{"role": "user", "content": "停机保号多少钱？"}]
    low_rerank = [Hit(CHUNKS[0], 0.2, reranked=True)]
    rrf_only = [Hit(CHUNKS[0], 0.03, reranked=False)]  # RRF 分数很小，但不代表不相关
    assert "没有找到" in llm.chat(messages, [], context=low_rerank).content
    assert "5 元/月" in llm.chat(messages, [], context=rrf_only).content


def test_offline_llm_routes_followup_to_previous_plan():
    llm = OfflineLLM()
    messages = [{"role": "user", "content": "5G畅享59元套餐包含多少流量？"},
                {"role": "assistant", "content": "60GB"},
                {"role": "user", "content": "如果我每月用75GB流量，要交多少钱？"}]
    reply = llm.chat(messages, [], context=[])
    assert reply.tool_calls[0].name == "calculate_fee"
    assert reply.tool_calls[0].arguments["plan"] == "5G畅享59元"
