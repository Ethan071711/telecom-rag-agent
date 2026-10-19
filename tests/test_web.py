"""网页演示的处理逻辑（不启动服务，不需要下载模型）：详情面板内容、多轮对话、清空、出错提示。"""

from telecom_rag.agent import TelecomAgent
from telecom_rag.llm import Reply, ToolCall
from telecom_rag.web import EMPTY_DETAILS, PENDING_DETAILS, chat_turn, render_details
from test_agent import CHUNKS, FakeRetriever, ScriptedLLM, settings


def test_render_details_shows_snippets_tools_and_latency():
    llm = ScriptedLLM([
        Reply(tool_calls=[ToolCall("c1", "calculate_fee", {"plan": "5G畅享59元", "data_gb": 75})]),
        Reply(content="本月合计 104 元 [2]。"),
    ])
    result = TelecomAgent(FakeRetriever(), llm, settings()).ask("59元套餐用了75GB流量要交多少钱？")
    details = render_details(result)
    assert f"{result.latency_s:.2f} 秒" in details
    assert "[2]《个人及家庭套餐资费总览》5G畅享59元" in details and CHUNKS[1].text in details
    assert "calculate_fee" in details and '"data_gb": 75' in details and '"合计(元)": 104' in details


def test_chat_turn_keeps_history_per_session_and_resets():
    retriever = FakeRetriever()
    llm = ScriptedLLM([Reply(content="可以办理停机保号 [1]。"), Reply(content="最长 12 个月 [1]。"),
                       Reply(content="保号费 5 元/月 [1]。")])
    base = TelecomAgent(retriever, llm, settings())

    pending, (session, messages, details) = chat_turn(base, None, "我想办理停机保号。")
    assert pending[1] == [{"role": "user", "content": "我想办理停机保号。"}] and pending[2] == PENDING_DETAILS
    *_, (session, messages, details) = chat_turn(base, session, "最长能保多久？")
    assert [m["content"] for m in messages[1::2]] == ["可以办理停机保号 [1]。", "最长 12 个月 [1]。"]
    assert retriever.queries[-1] == "我想办理停机保号。 最长能保多久？" and "追问" in details
    assert len(session.agent.history) == 4 and base.history == []  # 历史属于会话，共享的智能体不变

    # 示例问题每组的第一个问题会开启新对话
    *_, (session, messages, _) = chat_turn(base, session, "停机保号多少钱？", new_conversation=True)
    assert len(messages) == 2 and retriever.queries[-1] == "停机保号多少钱？"
    session.reset()
    assert session.agent.history == [] and session.messages == [] and session.details == EMPTY_DETAILS


def test_chat_turn_shows_error_and_keeps_working():
    class FlakyLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools, context=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("连接超时")
            return Reply(content="保号费 5 元/月 [1]。")

    base = TelecomAgent(FakeRetriever(), FlakyLLM(), settings())
    *_, (session, messages, details) = chat_turn(base, None, "停机保号多少钱？")
    assert "出错了" in messages[-1]["content"] and "RuntimeError: 连接超时" in details
    assert session.agent.history == []  # 出错的这一轮不进入对话历史
    *_, (session, messages, _) = chat_turn(base, session, "停机保号多少钱？")
    assert messages[-1]["content"] == "保号费 5 元/月 [1]。"
