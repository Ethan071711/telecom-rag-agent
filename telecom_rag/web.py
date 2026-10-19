"""网页演示（Gradio）：左边多轮对话，右边展示最近一轮回答引用的资料片段、调用的工具和耗时。

启动：python run.py web。组件由 run.py 的 build() 构建，和命令行完全相同，大模型后端仍由环境变量决定。
- 每个浏览器会话有自己的 TelecomAgent（各自保存对话历史），检索器、大模型和套餐数据在会话之间共享，
  所以刷新页面就是一段新对话；
- 所有提问排进同一个队列、一次只回答一个，避免本地大模型同时生成两段回答时显存不够。
"""

from __future__ import annotations

import json
import traceback

import gradio as gr

from .agent import AgentResult, TelecomAgent

TITLE = "中文运营商客服问答助手（RAG + 智能体）"
DISCLAIMER = ("**声明**：知识库是作者自行编写的演示资料，运营商“青禾通信”及其套餐、价格、规则都是**虚构**的，"
              "不代表中国联通或任何真实运营商的资费。")
PLACEHOLDER = ("**青禾通信智能客服**（虚构运营商）<br>"
               "可以问资费、流量包、携号转网、宽带、漫游、停机保号等业务，也可以让我按用量算话费、推荐套餐。")
EMPTY_DETAILS = "发送问题后，这里显示最近一轮回答引用的资料片段、调用的工具和耗时。"
PENDING_DETAILS = "正在检索资料、生成回答……"
TURN_MARKS = "①②③④⑤⑥⑦⑧⑨"
CSS = """
.disclaimer { border-left: 4px solid var(--color-accent); background: var(--color-accent-soft);
              padding: 4px 12px; border-radius: 4px; }
.details pre, .details pre code { white-space: pre-wrap; overflow-wrap: anywhere; }
.examples td:last-child, .examples td:last-child div { text-align: left; }
"""


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.splitlines())


def _code_block(text: str) -> str:
    """工具结果是 JSON 时缩进排版；search_knowledge 返回的是资料原文，按纯文本显示。"""
    try:
        return "```json\n" + json.dumps(json.loads(text), ensure_ascii=False, indent=2) + "\n```"
    except json.JSONDecodeError:
        return "```text\n" + text + "\n```"


def render_details(result: AgentResult, offline: bool = False) -> str:
    """右侧详情面板的 Markdown：耗时、检索 query、引用片段原文、工具调用的参数和结果。"""
    # 离线基线没有大模型，这里的“调用次数”是规则策略的决策次数
    calls = "规则决策" if offline else "模型调用"
    parts = [f"**耗时** {result.latency_s:.2f} 秒（含检索和 {result.llm_calls} 次{calls}）"]
    followup = "（追问，拼接了上一轮的问题）" if result.retrieval_query != result.question else ""
    parts.append(f"**检索 query**：{result.retrieval_query}{followup}")
    if result.refused:
        parts.append("**拒答**：知识库里没有找到相关信息")

    parts.append(f"#### 引用片段（检索到 {len(result.evidence)} 个，回答引用 {len(result.cited)} 个）")
    if not result.cited:
        parts.append("回答没有引用资料。")
    for n, hit in zip(result.cited, result.cited_hits):
        c = hit.chunk
        score = f" · 重排相关度 {hit.score:.2f}" if hit.reranked else ""
        parts.append(f"**[{n}]《{c.title}》{c.section}**{score}\n\n{_quote(c.text)}")

    parts.append(f"#### 工具调用（{len(result.tool_calls)} 次）")
    if not result.tool_calls:
        parts.append("本轮没有调用工具。")
    for i, call in enumerate(result.tool_calls, start=1):
        args = json.dumps(call["arguments"], ensure_ascii=False)
        parts.append(f"**{i}. `{call['name']}`**\n\n参数：\n```json\n{args}\n```\n\n结果：\n{_code_block(call['result'])}")
    return "\n\n".join(parts)


class ChatSession:
    """一个浏览器会话：自己的智能体（对话历史在 agent.history 里）、页面上的消息和最近一轮的详情。"""

    def __init__(self, base: TelecomAgent):
        # 检索器、大模型、套餐数据只读，所有会话共用；每个会话的 TelecomAgent 只多出一份对话历史
        self.agent = TelecomAgent(base.retriever, base.llm, base.settings, base.plans)
        self.offline = base.settings.llm_provider == "offline"
        self.messages: list[dict] = []
        self.details = EMPTY_DETAILS

    def reset(self) -> None:
        self.agent.reset()
        self.messages = []
        self.details = EMPTY_DETAILS

    def answer(self, question: str) -> str:
        try:
            result = self.agent.ask(question)
        except Exception as exc:  # 例如 API 密钥错误、网络中断：在页面上提示，而不是让页面一直转圈
            traceback.print_exc()
            self.details = f"**出错了**：`{type(exc).__name__}: {exc}`\n\n完整的错误信息见启动网页的终端。"
            return f"抱歉，出错了（{type(exc).__name__}），详细信息见终端。"
        self.details = render_details(result, self.offline)
        return result.answer


def chat_turn(base: TelecomAgent, session: ChatSession | None, question: str, new_conversation: bool = False):
    """处理一次提问的生成器：先显示问题，回答生成后再显示回答和详情。每次 yield (会话, 消息列表, 详情)。"""
    session = session or ChatSession(base)
    if new_conversation:
        session.reset()
    question = (question or "").strip()
    if question:
        session.messages.append({"role": "user", "content": question})
        session.details = PENDING_DETAILS
        yield session, list(session.messages), session.details
        session.messages.append({"role": "assistant", "content": session.answer(question)})
    yield session, list(session.messages), session.details


def create_demo(agent: TelecomAgent, examples: list[tuple[str, list[str]]] | None = None) -> gr.Blocks:
    """examples：[(类型, [第 1 轮问题, 第 2 轮追问, ...])]。每组的第一个问题开启新对话，后面的接着追问。"""
    settings = agent.settings
    rows, turns = [], []
    for label, group in examples or []:
        for i, question in enumerate(group):
            rows.append([label if len(group) == 1 else f"{label} {TURN_MARKS[i]}", question])
            turns.append((question, i == 0))

    backend = (f"大模型后端 `{settings.describe()}` · 向量模型 `{settings.embed_model}` · "
               f"重排模型 `{settings.rerank_model or '不启用'}`")
    if settings.llm_provider == "offline":
        backend += "  \n未配置大模型，当前是离线规则基线（抽取式回答 + 规则路由工具），切换方法见 .env.example。"

    with gr.Blocks(title="青禾通信智能客服演示", analytics_enabled=False) as demo:
        session = gr.State(None)
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DISCLAIMER, elem_classes=["disclaimer"])
        gr.Markdown(backend)
        with gr.Row():
            with gr.Column(scale=3):
                # 去掉默认的 share 按钮（分享到 Hugging Face），只保留复制
                chatbot = gr.Chatbot(label="对话", height=520, placeholder=PLACEHOLDER, buttons=["copy", "copy_all"])
                textbox = gr.Textbox(placeholder="输入问题，按回车发送", show_label=False, submit_btn="发送",
                                     autofocus=True)
                clear = gr.Button("清空对话")
                example_table = None
                if rows:
                    example_table = gr.Dataset(
                        label="示例问题（点击直接发送；每组示例开启新对话，多轮追问请按 ①→② 的顺序点击）",
                        components=["textbox", "textbox"], headers=["类型", "问题"], samples=rows, type="index",
                        elem_classes=["examples"])
            with gr.Column(scale=2):
                gr.Markdown("### 最近一轮回答")
                details = gr.Markdown(EMPTY_DETAILS, line_breaks=True, elem_classes=["details"])

        def on_submit(state, text):
            for outputs in chat_turn(agent, state, text):
                yield *outputs, ""  # 第一次 yield 时就清空输入框

        def on_example(state, index):
            question, new_conversation = turns[index]
            yield from chat_turn(agent, state, question, new_conversation)

        def on_clear(state):
            if state is not None:
                state.reset()  # 调用 agent.reset()，清空这个会话的对话历史
            return state, [], EMPTY_DETAILS

        # 同一个 concurrency_id、上限 1：所有会话的提问和清空排成一队，见模块说明
        serial = {"concurrency_id": "agent", "concurrency_limit": 1}
        textbox.submit(on_submit, [session, textbox], [session, chatbot, details, textbox], **serial)
        # 对话框右上角自带的清空图标也要清空智能体的对话历史，否则页面空了、模型却还记得上文
        gr.on([clear.click, chatbot.clear], on_clear, [session], [session, chatbot, details], **serial)
        if example_table is not None:
            example_table.click(on_example, [session, example_table], [session, chatbot, details], **serial)
    return demo


def launch(agent: TelecomAgent, examples: list[tuple[str, list[str]]] | None = None,
           port: int | None = None) -> None:
    """启动网页服务并阻塞，Ctrl+C 退出。port 为 None 时 Gradio 从 7860 开始找空闲端口。"""
    demo = create_demo(agent, examples)
    demo.queue().launch(server_port=port, css=CSS, footer_links=["gradio", "settings"])
