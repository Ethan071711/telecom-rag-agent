"""中文向量化（bge-small-zh）+ FAISS 本地向量库：构建、保存、加载。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np

from .config import DOCS_DIR, STORAGE_DIR, Settings
from .documents import Chunk, chunk_documents, corpus_fingerprint, load_documents

# bge 系列推荐给“短查询找长段落”的查询加上这句指令，文档侧不加
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype="float32")

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_docs([BGE_QUERY_INSTRUCTION + text])


class VectorStore:
    """向量已归一化，用内积索引（IndexFlatIP）等价于余弦相似度。几百个片段用精确检索就够了。"""

    def __init__(self, chunks: list[Chunk], index: faiss.Index, meta: dict):
        self.chunks = chunks
        self.index = index
        self.meta = meta

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, meta: dict) -> "VectorStore":
        vectors = embedder.encode_docs([c.embed_text for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(chunks, index, {**meta, "dim": int(vectors.shape[1]), "num_chunks": len(chunks)})

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # faiss.write_index 在 Windows 上不支持中文路径，所以先序列化成字节，再交给 Python 写文件
        (out_dir / "index.faiss").write_bytes(faiss.serialize_index(self.index).tobytes())
        with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        (out_dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path) -> "VectorStore":
        raw = np.frombuffer((out_dir / "index.faiss").read_bytes(), dtype="uint8")
        index = faiss.deserialize_index(raw)
        with open(out_dir / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(chunks, index, meta)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores, ids = self.index.search(query_vec, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def ensure_index(settings: Settings, embedder: Embedder, rebuild: bool = False,
                 docs_dir: Path = DOCS_DIR, out_dir: Path = STORAGE_DIR) -> VectorStore:
    """索引不存在或文档有变化时自动重建，否则直接加载。"""
    fingerprint = corpus_fingerprint(docs_dir, settings.embed_model, str(settings.chunk_size))
    if not rebuild and (out_dir / "meta.json").exists():
        store = VectorStore.load(out_dir)
        if store.meta.get("fingerprint") == fingerprint:
            return store
        print("[index] 检测到文档或配置有变化，重建索引……")
    t0 = time.time()
    docs = load_documents(docs_dir)
    chunks = chunk_documents(docs, settings.chunk_size)
    meta = {"fingerprint": fingerprint, "embed_model": settings.embed_model,
            "chunk_size": settings.chunk_size, "num_docs": len(docs)}
    store = VectorStore.build(chunks, embedder, meta)
    store.save(out_dir)
    print(f"[index] {len(docs)} 篇文档 -> {len(chunks)} 个片段，向量维度 {store.meta['dim']}，"
          f"耗时 {time.time() - t0:.1f}s，已保存到 {out_dir.name}/")
    return store
