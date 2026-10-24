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


def test_tool_loop_uses_computed_fee_instead_of_model_paraphrase():
    llm = ScriptedLLM([
        Reply(tool_calls=[ToolCall("c1", "calculate_fee", {"plan": "5G畅享59元", "data_gb": 75, "voice_minutes": 300})]),
        Reply(content="本月合计 999 元 [2]。"),
    ])
    agent = TelecomAgent(FakeRetriever(), llm, settings())
    result = agent.ask("59元套餐用了75GB流量、300分钟要交多少钱？")
    assert '本月费用合计 104 元' in result.answer and '999' not in result.answer
    assert result.tool_calls[0]["name"] == "calculate_fee"
    assert '"合计(元)": 104' in result.tool_calls[0]["result"]
    assert result.grounding['answer_source']=='deterministic_fee_tool'
    assert result.cited_hits==[]
    # 第二次调用时，模型能看到 assistant 的工具调用和工具结果
    roles = [m["role"] for m in llm.seen[1]]
    assert roles[-2:] == ["assistant", "tool"]
    assert agent.history[-1]["content"] == result.answer


def test_followup_question_uses_previous_turn_for_retrieval():
    retriever = FakeRetriever()
    llm = ScriptedLLM([Reply(content="可以办理停机保号 [1]。"), Reply(content="最长 12 个月 [1]。")])
    agent = TelecomAgent(retriever, llm, settings())
    agent.ask("我想办理停机保号。")
    agent.ask("最长能保多久？")
    assert retriever.queries[-1] == "我想办理停机保号。 最长能保多久？"


def test_grade_flags_unsupported_numbers():
    from dataclasses import replace
    llm = ScriptedLLM([Reply(content="保号费 5 元/月，最长 12 个月 [1]。")])
    agent = TelecomAgent(FakeRetriever(), llm, settings())
    result = replace(agent.ask("停机保号多少钱？"), answer="保号费 8 元/月，最长 12 个月 [1]。")
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


def test_unsupported_answer_is_corrected_before_history_or_display():
    llm = ScriptedLLM([Reply(content="保号费 8 元/月 [1]。"), Reply(content="保号费 5 元/月 [1]。")])
    agent = TelecomAgent(FakeRetriever(), llm, settings())
    result = agent.ask("停机保号多少钱？")
    assert result.answer == "保号费 5 元/月 [1]。"
    assert result.grounding['status'] == 'repaired' and result.llm_calls == 2
    assert '8 元' not in agent.history[-1]['content']


def test_repeated_bad_revision_falls_back_to_exact_source():
    llm = ScriptedLLM([Reply(content="保号费 8 元/月 [1]。"), Reply(content="保号费 9 元/月 [1]。")])
    result = TelecomAgent(FakeRetriever(), llm, settings()).ask("停机保号多少钱？")
    assert result.grounding['status'] == 'extractive_fallback'
    assert CHUNKS[0].text in result.answer and '8 元' not in result.answer and '9 元' not in result.answer


def test_invalid_citation_and_failed_revision_do_not_escape():
    class BrokenRepair(ScriptedLLM):
        def chat(self, messages, tools, context=None):
            if self.replies:
                return self.replies.pop()
            raise TimeoutError('repair unavailable')
    result = TelecomAgent(FakeRetriever(), BrokenRepair([Reply(content='保号费 5 元 [999]。')]), settings()).ask('保号费？')
    assert result.refused and not result.cited
    assert result.grounding['status'] == 'extractive_fallback'


def test_failed_tool_result_cannot_authorize_an_invented_bill():
    llm = ScriptedLLM([
        Reply(tool_calls=[ToolCall('x', 'calculate_fee', {'plan': '59元', 'data_gb': -3})]),
        Reply(content='合计 88 元。'), Reply(content='合计 89 元。')])
    result = TelecomAgent(FakeRetriever(), llm, settings()).ask('我的流量是负数，多少钱？')
    assert result.refused and '88' not in result.answer and '89' not in result.answer


def test_failed_answer_reads_other_sections_of_its_source_before_revision():
    class Expandable(FakeRetriever):
        def expand_sources(self, hits):
            assert hits[0].chunk.doc_id == CHUNKS[0].doc_id
            return [Hit(Chunk('09_suspension#1','09_suspension','停机保号','恢复',
                              '恢复业务需要本人确认。'),0)]
    llm=ScriptedLLM([Reply(content='保号费8元[1]。'),Reply(content='保号费5元[1]，恢复业务需要本人确认[3]。')])
    result=TelecomAgent(Expandable(),llm,settings()).ask('保号费和恢复要求？')
    assert result.grounding['source_chunks_added']==1
    assert '恢复业务需要本人确认' in llm.seen[-1][0]['content']
    assert result.cited==[1,3]
