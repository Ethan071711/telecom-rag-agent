"""加载 Markdown 知识库并切块。

切块策略（面试时可以这样讲）：
1. 先按 Markdown 标题拆成小节，保证一个片段只讲一件事；
2. 小节太长时按段落打包，单个段落仍然太长时：表格按行拆分并重复表头，普通文字按句号拆分；
3. 每个片段在向量化时都带上“文档标题 + 小节路径”，补足上下文（例如“副卡 > 共享规则”）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SENTENCE_END_RE = re.compile(r"(?<=[。！？；])")


@dataclass
class Document:
    doc_id: str
    title: str
    category: str
    source: str
    body: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str

    @property
    def embed_text(self) -> str:
        """用于向量化和 BM25 的文本：标题 + 小节 + 正文。"""
        return f"{self.title} {self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """解析文件开头 --- 包起来的 key: value 元信息。"""
    meta: dict[str, str] = {}
    lines = raw.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                for line in lines[1:i]:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        meta[key.strip()] = value.strip()
                return meta, "\n".join(lines[i + 1:])
    return meta, raw


def load_documents(docs_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(docs_dir.glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        docs.append(Document(
            doc_id=path.stem,
            title=meta.get("title", path.stem),
            category=meta.get("category", ""),
            source=meta.get("source", "未注明"),
            body=body.strip(),
        ))
    if not docs:
        raise FileNotFoundError(f"{docs_dir} 下没有找到 .md 文档")
    return docs


def split_sections(body: str) -> list[tuple[str, str]]:
    """按标题拆分，返回 [(小节路径, 小节正文)]。一级标题就是文档标题，不放进路径。"""
    sections: list[tuple[str, str]] = []
    heading_path: dict[int, str] = {}
    current_path, buf = "", []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append((current_path or "概述", text))
        buf.clear()

    for line in body.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading_path = {k: v for k, v in heading_path.items() if k < level}
            heading_path[level] = match.group(2).strip()
            current_path = " > ".join(heading_path[k] for k in sorted(heading_path) if k > 1)
        else:
            buf.append(line)
    flush()
    return sections


def _split_table(block: str, size: int) -> list[str]:
    """表格按行拆分，每一块都重复表头和分隔行，避免丢失列含义。"""
    lines = block.splitlines()
    header, rows = lines[:2], lines[2:]
    pieces, current = [], list(header)
    for row in rows:
        if len(current) > 2 and len("\n".join(current + [row])) > size:
            pieces.append("\n".join(current))
            current = list(header)
        current.append(row)
    pieces.append("\n".join(current))
    return pieces


def _split_long_text(block: str, size: int) -> list[str]:
    sentences = [s for s in SENTENCE_END_RE.split(block) if s.strip()]
    pieces, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > size:
            pieces.append(current)
            current = ""
        current += sentence
    if current:
        pieces.append(current)
    return pieces


def _pack_blocks(text: str, size: int) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    pieces, buf = [], ""
    for block in blocks:
        if len(block) > size:
            if buf:
                pieces.append(buf)
                buf = ""
            is_table = block.lstrip().startswith("|")
            pieces.extend(_split_table(block, size) if is_table else _split_long_text(block, size))
        elif buf and len(buf) + len(block) + 1 > size:
            pieces.append(buf)
            buf = block
        else:
            buf = f"{buf}\n{block}" if buf else block
    if buf:
        pieces.append(buf)
    return pieces


def chunk_documents(docs: list[Document], chunk_size: int = 350) -> list[Chunk]:
    chunks = []
    for doc in docs:
        n = 0
        for section, text in split_sections(doc.body):
            for piece in _pack_blocks(text, chunk_size):
                chunks.append(Chunk(f"{doc.doc_id}#{n}", doc.doc_id, doc.title, section, piece))
                n += 1
    return chunks


def corpus_fingerprint(docs_dir: Path, *extra: str) -> str:
    """文档内容 + 切块参数 + 模型名的哈希，用来判断索引是否过期。"""
    h = hashlib.sha256()
    for path in sorted(docs_dir.glob("*.md")):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    for item in extra:
        h.update(item.encode("utf-8"))
    return h.hexdigest()[:16]
