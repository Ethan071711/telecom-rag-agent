"""混合检索 + 重排。

流程：BM25（jieba 分词，擅长精确匹配“59元”“LOS灯”这类关键词）和向量检索（擅长同义改写）
各取前 N 个候选 -> RRF（倒数排名融合）合并 -> 交叉编码器 bge-reranker 逐条打分重排 -> 取前 k 个。
重排分数低于阈值的片段直接丢弃；一个都不剩时说明知识库里没有相关内容，交给上层拒答。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

from .config import USERDICT_PATH, Settings
from .documents import Chunk
from .index import Embedder, VectorStore

jieba.setLogLevel(logging.WARNING)

STOPWORDS = set(
    "的 了 吗 呢 吧 啊 呀 是 我 我们 你 你们 您 他 她 它 这 那 这个 那个 有 在 和 与 及 或 就 都 也 还 又 "
    "怎么 怎么样 怎样 什么 多少 多久 几 哪个 哪些 可以 能 能不能 会 要 需要 想 请问 一下 如果 一个 现在 么".split()
)
TOKEN_RE = re.compile(r"[\w.]+", re.UNICODE)
RRF_K = 60

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float  # 最后一个阶段的分数：重排模式是 0~1 的相关度，其他模式是各自的原始分数
    reranked: bool = False  # score 是否为重排相关度（只有它能和阈值比较）


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.settings = settings
        if USERDICT_PATH.exists():
            jieba.load_userdict(str(USERDICT_PATH))
        self.bm25 = BM25Okapi([self.tokenize(c.embed_text) for c in store.chunks])
        self.reranker = None
        if settings.rerank_model:
            import torch
            from sentence_transformers import CrossEncoder

            device = settings.retrieval_device
            on_gpu = torch.cuda.is_available() and device != "cpu"
            # 在 GPU 上用半精度：显存减半（约 0.5GB），分数与 fp32 基本一致
            kwargs = {"model_kwargs": {"dtype": torch.float16}} if on_gpu else {}
            self.reranker = CrossEncoder(settings.rerank_model, max_length=512, device=device, **kwargs)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        tokens = []
        for tok in jieba.lcut_for_search(text.lower()):
            tok = tok.strip()
            if tok and tok not in STOPWORDS and TOKEN_RE.fullmatch(tok):
                tokens.append(tok)
        return tokens

    # ---- 三种召回 ----
    def dense(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.store.search(self.embedder.encode_query(query), k)

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(self.tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in order if scores[i] > 0]

    def hybrid(self, query: str, k: int) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for ranked in (self.dense(query, k), self.sparse(query, k)):
            for rank, (idx, _) in enumerate(ranked):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def rerank(self, query: str, candidates: list[int]) -> list[tuple[int, float]]:
        if not candidates:
            return []
        pairs = [(query, self.store.chunks[i].embed_text) for i in candidates]
        raw = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        # 统一成 0~1：如果模型输出的是 logits，就过一下 sigmoid
        scores = [float(s) for s in raw]
        if any(s < 0 or s > 1 for s in scores):
            scores = [1 / (1 + math.exp(-s)) for s in scores]
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # ---- 对外接口 ----
    def search(self, query: str, top_k: int | None = None, mode: str = "hybrid_rerank",
               apply_threshold: bool = True) -> list[Hit]:
        top_k = top_k or self.settings.top_k
        n = self.settings.candidates
        reranked = mode == "hybrid_rerank" and self.reranker is not None
        if mode == "dense":
            ranked = self.dense(query, n)
        elif mode == "bm25":
            ranked = self.sparse(query, n)
        elif mode == "hybrid" or (mode == "hybrid_rerank" and self.reranker is None):
            ranked = self.hybrid(query, n)
        elif mode == "hybrid_rerank":
            ranked = self.rerank(query, [i for i, _ in self.hybrid(query, n)])
            if apply_threshold:
                ranked = [(i, s) for i, s in ranked if s >= self.settings.min_rerank_score]
        else:
            raise ValueError(f"未知检索模式 {mode}，可选：{MODES}")
        return [Hit(self.store.chunks[i], s, reranked) for i, s in ranked[:top_k]]
