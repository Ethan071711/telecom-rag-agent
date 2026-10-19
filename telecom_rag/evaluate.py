"""评测：检索消融 + 端到端问答。

端到端指标（全部可复现、不依赖大模型打分）：
- 答案准确率：可回答的题，回答包含全部关键事实（每个关键事实允许多种写法）且没有拒答；
              知识库外的题，正确拒答才算对。
- 引用正确率：有标准出处的题，回答引用的片段里至少有一个来自标准出处文档。另报引用精确率（引用中来自标准出处的比例）。
- 工具调用正确率：需要工具的题，调用了预期的工具。
- 无依据回答比例：① 知识库外的题没有拒答；② 既没引用资料也没调用工具；③ 回答里带单位的数字
                  （如 189元、120天）在给模型看过的资料、工具结果和问题里都找不到。满足任一条即记为无依据。
"""

from __future__ import annotations

import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from .config import EVAL_PATH, RESULTS_DIR, Settings
from .retriever import Retriever
from .textutil import normalize, numeric_claims

RETRIEVAL_MODES = {"dense": "向量检索", "bm25": "BM25", "hybrid": "混合(RRF)", "hybrid_rerank": "混合+重排"}
TYPE_NAMES = {"kb": "知识问答", "tool": "工具计算", "multi_turn": "多轮对话", "unanswerable": "知识库外"}


def load_eval_set(path: Path = EVAL_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def turns_of(item: dict) -> list[str]:
    return item.get("turns") or [item["question"]]


# ---------------- 检索消融 ----------------
def retrieval_query_for(item: dict) -> str:
    """多轮题用“上一轮 + 本轮”作为检索 query，与智能体的做法一致。"""
    turns = turns_of(item)
    return " ".join(turns[-2:]) if len(turns) > 1 else turns[0]


def evaluate_retrieval(retriever: Retriever, items: list[dict], k: int = 5) -> dict:
    graded = [it for it in items if it.get("gold_docs")]
    report: dict = {"num_questions": len(graded), "k": k, "modes": {}}
    for mode in RETRIEVAL_MODES:
        r1, r3, rk, mrr = [], [], [], []
        for item in graded:
            hits = retriever.search(retrieval_query_for(item), top_k=k, mode=mode, apply_threshold=False)
            docs = [h.chunk.doc_id for h in hits]
            gold = set(item["gold_docs"])
            rank = next((i + 1 for i, d in enumerate(docs) if d in gold), None)
            r1.append(rank == 1)
            r3.append(rank is not None and rank <= 3)
            rk.append(rank is not None)
            mrr.append(1 / rank if rank else 0.0)
        report["modes"][mode] = {
            "Recall@1": statistics.mean(r1), "Recall@3": statistics.mean(r3),
            f"Recall@{k}": statistics.mean(rk), "MRR": statistics.mean(mrr),
        }
    # 知识库外问题的最高重排分数，用来解释拒答阈值
    if retriever.reranker is not None:
        top = {}
        for item in items:
            hits = retriever.search(retrieval_query_for(item), top_k=1, apply_threshold=False)
            top.setdefault(item["type"], []).append(round(hits[0].score, 3) if hits else 0.0)
        report["top_rerank_score_by_type"] = {t: {"min": min(v), "median": round(statistics.median(v), 3),
                                                  "max": max(v)} for t, v in top.items()}
    return report


# ---------------- 端到端 ----------------
def contains_all(answer: str, groups: list[list[str]]) -> bool:
    text = normalize(answer)
    return all(any(normalize(alt) in text for alt in group) for group in groups)


def grade(item: dict, result) -> dict:
    gold_docs = set(item.get("gold_docs") or [])
    expected_tools = set(item.get("expected_tools") or [])
    cited_docs = [result.evidence[n - 1].chunk.doc_id for n in result.cited]
    called = [t["name"] for t in result.tool_calls]
    answerable = item["type"] != "unanswerable"

    correct = (not result.refused and contains_all(result.answer, item.get("must_include", []))) if answerable \
        else result.refused

    citation_hit = citation_precision = None
    if answerable and gold_docs:
        citation_hit = bool(set(cited_docs) & gold_docs)
        citation_precision = sum(d in gold_docs for d in cited_docs) / len(cited_docs) if cited_docs else 0.0

    tool_ok = bool(expected_tools & set(called)) if expected_tools else None

    reason = ""
    if not answerable and not result.refused:
        reason = "知识库外的问题没有拒答"
    elif answerable and not result.refused:
        if not result.cited and not called:
            reason = "没有引用资料也没有调用工具"
        else:
            evidence = normalize(" ".join([h.chunk.text for h in result.evidence]
                                          + [t["result"] for t in result.tool_calls] + turns_of(item)))
            missing = [n for n in numeric_claims(result.answer) if n not in evidence]
            if missing:
                reason = f"数字在资料中找不到：{'、'.join(missing[:5])}"
    return {
        "id": item["id"], "type": item["type"], "turns": turns_of(item), "answer": result.answer,
        "reference": item.get("answer", ""), "correct": correct, "refused": result.refused,
        "cited_docs": cited_docs, "gold_docs": sorted(gold_docs), "citation_hit": citation_hit,
        "citation_precision": citation_precision, "tools_called": called, "expected_tools": sorted(expected_tools),
        "tool_ok": tool_ok, "unsupported": bool(reason), "unsupported_reason": reason,
        "retrieval_query": result.retrieval_query, "latency_s": round(result.latency_s, 2),
        "llm_calls": result.llm_calls,
    }


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(details: list[dict]) -> dict:
    metrics = {
        "num_questions": len(details),
        "answer_accuracy": _mean(d["correct"] for d in details),
        "citation_accuracy": _mean(d["citation_hit"] for d in details),
        "citation_precision": _mean(d["citation_precision"] for d in details),
        "tool_accuracy": _mean(d["tool_ok"] for d in details),
        "unsupported_rate": _mean(d["unsupported"] for d in details),
        "refusal_accuracy": _mean(d["refused"] for d in details if d["type"] == "unanswerable"),
        "false_refusal_rate": _mean(d["refused"] for d in details if d["type"] != "unanswerable"),
        "avg_latency_s": _mean(d["latency_s"] for d in details),
        "avg_llm_calls": _mean(d["llm_calls"] for d in details),
        "accuracy_by_type": {},
    }
    for t in TYPE_NAMES:
        subset = [d["correct"] for d in details if d["type"] == t]
        if subset:
            metrics["accuracy_by_type"][t] = {"n": len(subset), "accuracy": statistics.mean(subset)}
    return metrics


def run_e2e(agent, items: list[dict], verbose: bool = True) -> list[dict]:
    details = []
    for i, item in enumerate(items, 1):
        agent.reset()
        result = None
        for turn in turns_of(item):  # 多轮题按顺序提问，只评最后一轮的回答
            result = agent.ask(turn)
        d = grade(item, result)
        details.append(d)
        if verbose:
            flag = "对" if d["correct"] else "错"
            extra = f" | 无依据：{d['unsupported_reason']}" if d["unsupported"] else ""
            print(f"  [{i:>2}/{len(items)}] {item['id']:<6} {flag} {d['latency_s']:>5.1f}s x{d['llm_calls']} "
                  f"{d['answer'][:40].replace(chr(10), ' ')}{extra}")
    return details


# ---------------- 结果写入 ----------------
def slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", text).strip("-").lower()


def save_e2e(settings: Settings, details: list[dict], metrics: dict, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if settings.rerank_model else "_norerank"  # 关闭重排的消融结果单独保存，不覆盖默认配置
    path = results_dir / f"e2e_{slug(settings.llm_provider)}_{slug(settings.llm_model.split('/')[-1])}{suffix}.json"
    label = settings.describe() + ("" if settings.rerank_model else "（关闭重排）")
    payload = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "llm": label, "embed_model": settings.embed_model, "rerank_model": settings.rerank_model,
        "top_k": settings.top_k, "min_rerank_score": settings.min_rerank_score,
        "metrics": metrics, "details": details,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_retrieval(report: dict, settings: Settings, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "retrieval_ablation.json"
    payload = {"run_at": datetime.now().isoformat(timespec="seconds"), "embed_model": settings.embed_model,
               "rerank_model": settings.rerank_model, **report}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def write_summary(results_dir: Path = RESULTS_DIR) -> Path:
    """汇总 results/ 下所有评测结果，生成 summary.md（README 里的表格就是从这里抄的）。"""
    lines = ["# 评测结果汇总", "", f"生成时间：{datetime.now():%Y-%m-%d %H:%M}", ""]
    retrieval = results_dir / "retrieval_ablation.json"
    if retrieval.exists():
        rep = json.loads(retrieval.read_text(encoding="utf-8"))
        k = rep["k"]
        lines += [f"## 检索消融（{rep['num_questions']} 道有标准出处的题，文档级命中）", "",
                  f"| 检索方式 | Recall@1 | Recall@3 | Recall@{k} | MRR |", "| --- | --- | --- | --- | --- |"]
        for mode, m in rep["modes"].items():
            lines.append(f"| {RETRIEVAL_MODES[mode]} | {_pct(m['Recall@1'])} | {_pct(m['Recall@3'])} | "
                         f"{_pct(m[f'Recall@{k}'])} | {m['MRR']:.3f} |")
        if "top_rerank_score_by_type" in rep:
            lines += ["", "各类问题的最高重排分数（min / 中位数 / max）："]
            for t, s in rep["top_rerank_score_by_type"].items():
                lines.append(f"- {TYPE_NAMES.get(t, t)}：{s['min']} / {s['median']} / {s['max']}")
        lines.append("")

    runs = sorted(results_dir.glob("e2e_*.json"))
    if runs:
        lines += ["## 端到端问答", "",
                  "| 后端 | 题数 | 答案准确率 | 引用正确率 | 引用精确率 | 工具调用正确率 | 无依据回答比例 | 拒答正确率 | 误拒率 | 平均耗时 | 平均模型调用次数 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        by_type_rows = []
        for path in runs:
            run = json.loads(path.read_text(encoding="utf-8"))
            m = run["metrics"]
            lines.append(f"| {run['llm']} | {m['num_questions']} | {_pct(m['answer_accuracy'])} | "
                         f"{_pct(m['citation_accuracy'])} | {_pct(m['citation_precision'])} | {_pct(m['tool_accuracy'])} | "
                         f"{_pct(m['unsupported_rate'])} | {_pct(m['refusal_accuracy'])} | "
                         f"{_pct(m['false_refusal_rate'])} | {m['avg_latency_s']:.2f}s | "
                         f"{m.get('avg_llm_calls') or 0:.2f} |")
            by_type_rows.append((run["llm"], m["accuracy_by_type"]))
        lines += ["", "分题型答案准确率：", "", "| 后端 | " + " | ".join(TYPE_NAMES.values()) + " |",
                  "| --- |" + " --- |" * len(TYPE_NAMES)]
        for llm, by_type in by_type_rows:
            cells = [f"{_pct(by_type[t]['accuracy'])} (n={by_type[t]['n']})" if t in by_type else "-" for t in TYPE_NAMES]
            lines.append(f"| {llm} | " + " | ".join(cells) + " |")
    path = results_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0
