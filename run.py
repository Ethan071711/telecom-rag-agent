"""项目入口。

  python run.py                 一条命令跑通：建索引 -> 检索消融 -> 演示问答 -> 端到端评测，结果写入 results/
  python run.py index [--rebuild]
  python run.py ask "问题"       单轮问答
  python run.py chat            多轮对话（/reset 清空上下文，/exit 退出）
  python run.py web [--port N]  网页演示（Gradio）：多轮对话，右侧显示引用片段、工具调用和耗时
  python run.py eval [--limit N] [--retrieval-only]

大模型后端由环境变量（或 .env）决定，见 .env.example；什么都不配时使用离线规则基线。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows 上 faiss 和 torch 各带一份 OpenMP 时避免报错
warnings.filterwarnings("ignore", category=SyntaxWarning)  # jieba 源码在 Python 3.12+ 下的转义序列警告
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

# 演示问答和网页示例共用：(类型, [第 1 轮问题, 追问, ...])，每组从新对话开始
DEMO_QUESTIONS = [
    ("知识问答", ["携号转网对在网时长有什么要求？"]),
    ("费用计算", ["我用的是5G畅享39元套餐，这个月用了70GB流量、打了100分钟电话，要交多少钱？"]),
    ("多轮追问", ["青春校园卡每个月多少钱？", "那毕业以后会怎么样？"]),
]


def console_safe() -> None:
    """Windows 控制台被重定向时编码可能是 GBK，遇到无法编码的字符用 ? 代替，不让程序崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def build(need_agent: bool = True, rebuild: bool = False):
    from telecom_rag.config import Settings

    settings = Settings.from_env()  # 先读 .env，HF_ENDPOINT 等变量要在导入 huggingface 相关库之前生效

    from telecom_rag.index import Embedder, ensure_index
    from telecom_rag.retriever import Retriever

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except ImportError:
        pass

    print(f"[config] 大模型后端：{settings.describe()} | 向量模型：{settings.embed_model} | "
          f"重排模型：{settings.rerank_model or '不启用'}")
    if settings.llm_provider == "offline":
        print("[config] 未配置大模型，使用离线规则基线（抽取式回答 + 规则路由工具）。配置方法见 .env.example")
    embedder = Embedder(settings.embed_model, device=settings.retrieval_device)
    store = ensure_index(settings, embedder, rebuild=rebuild)
    retriever = Retriever(store, embedder, settings)
    if not need_agent:
        return settings, retriever, None

    from telecom_rag.agent import TelecomAgent
    from telecom_rag.llm import create_llm
    from telecom_rag.tools import PlanBook

    plans = PlanBook()
    agent = TelecomAgent(retriever, create_llm(settings, plans), settings, plans)
    return settings, retriever, agent


def cmd_demo(agent) -> None:
    print("\n========== 演示问答 ==========")
    for _, turns in DEMO_QUESTIONS:
        agent.reset()
        for q in turns:
            result = agent.ask(q)
            print(f"\n用户：{q}\n助手：{result.format()}\n（{result.latency_s:.1f}s）")


def cmd_eval(settings, retriever, agent, limit: int | None, retrieval_only: bool) -> None:
    from telecom_rag import evaluate as ev

    items = ev.load_eval_set()
    if retriever.reranker is None:
        print("\n（已关闭重排，跳过检索消融，保留 results/retrieval_ablation.json 中的完整结果）")
    else:
        print(f"\n========== 检索消融（{len([i for i in items if i.get('gold_docs')])} 题） ==========")
        report = ev.evaluate_retrieval(retriever, items)
        for mode, m in report["modes"].items():
            print(f"  {ev.RETRIEVAL_MODES[mode]:<8} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
        ev.save_retrieval(report, settings)
    if not retrieval_only:
        subset = items[:limit] if limit else items
        print(f"\n========== 端到端评测：{settings.describe()}，{len(subset)} 题 ==========")
        details = ev.run_e2e(agent, subset)
        metrics = ev.summarize(details)
        path = ev.save_e2e(settings, details, metrics)
        print(f"\n  答案准确率 {ev._pct(metrics['answer_accuracy'])} | 引用来源命中率 {ev._pct(metrics['citation_accuracy'])}"
              f" | 工具调用正确率 {ev._pct(metrics['tool_accuracy'])} | 无依据回答比例 {ev._pct(metrics['unsupported_rate'])}")
        print(f"  明细已保存：results/{path.name}")
    summary = ev.write_summary()
    print(f"  汇总表已更新：results/{summary.name}")


def cmd_chat(agent) -> None:
    print("\n多轮对话已开始（/reset 清空上下文，/exit 退出）")
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in {"/exit", "/quit", "exit", "quit"}:
            break
        if q == "/reset":
            agent.reset()
            print("（上下文已清空）")
            continue
        result = agent.ask(q)
        print(f"助手：{result.format()}\n（{result.latency_s:.1f}s）")


def main() -> None:
    console_safe()
    parser = argparse.ArgumentParser(description="中文运营商客服问答助手（RAG + 智能体）")
    sub = parser.add_subparsers(dest="cmd")
    p_index = sub.add_parser("index", help="构建向量索引")
    p_index.add_argument("--rebuild", action="store_true")
    p_ask = sub.add_parser("ask", help="单轮问答")
    p_ask.add_argument("question")
    sub.add_parser("chat", help="多轮对话")
    p_web = sub.add_parser("web", help="网页演示（Gradio）")
    p_web.add_argument("--port", type=int, help="端口，默认从 7860 开始找空闲端口")
    p_eval = sub.add_parser("eval", help="评测")
    p_eval.add_argument("--limit", type=int)
    p_eval.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    if args.cmd == "index":
        build(need_agent=False, rebuild=args.rebuild)
    elif args.cmd == "ask":
        _, _, agent = build()
        result = agent.ask(args.question)
        print(f"\n助手：{result.format()}\n（{result.latency_s:.1f}s）")
    elif args.cmd == "chat":
        _, _, agent = build()
        cmd_chat(agent)
    elif args.cmd == "web":
        _, _, agent = build()
        from telecom_rag.web import launch  # 在 build() 读完 .env 之后再导入 gradio（它会导入 huggingface_hub）

        launch(agent, DEMO_QUESTIONS, args.port)
    elif args.cmd == "eval":
        settings, retriever, agent = build(need_agent=not args.retrieval_only)
        cmd_eval(settings, retriever, agent, args.limit, args.retrieval_only)
    else:
        settings, retriever, agent = build()
        cmd_demo(agent)
        cmd_eval(settings, retriever, agent, limit=None, retrieval_only=False)


if __name__ == "__main__":
    main()
