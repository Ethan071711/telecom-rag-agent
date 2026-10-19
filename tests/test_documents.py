from telecom_rag.config import DOCS_DIR
from telecom_rag.documents import (Document, chunk_documents, load_documents, parse_front_matter,
                                   split_sections)


def test_front_matter():
    meta, body = parse_front_matter("---\ntitle: 测试\nsource: 自行编写\n---\n# 标题\n正文")
    assert meta == {"title": "测试", "source": "自行编写"}
    assert body.strip().startswith("# 标题")


def test_split_sections_keeps_heading_path():
    sections = split_sections("# 文档\n开头\n## 一、条件\n内容A\n### 细则\n内容B\n## 二、流程\n内容C")
    assert sections == [("概述", "开头"), ("一、条件", "内容A"), ("一、条件 > 细则", "内容B"), ("二、流程", "内容C")]


def test_long_table_is_split_with_header():
    header = "| 名称 | 价格 |\n| --- | --- |"
    rows = "\n".join(f"| 套餐{i}号很长的名字用来凑字数 | {i}元 |" for i in range(30))
    doc = Document("t", "表格", "", "", f"# 表格\n{header}\n{rows}")
    chunks = chunk_documents([doc], chunk_size=200)
    assert len(chunks) > 1
    assert all(c.text.startswith("| 名称 | 价格 |") for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)


def test_knowledge_base_loads():
    docs = load_documents(DOCS_DIR)
    assert 10 <= len(docs) <= 30
    assert all(d.source and d.title for d in docs)
    chunks = chunk_documents(docs, 350)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert max(len(c.text) for c in chunks) <= 350
