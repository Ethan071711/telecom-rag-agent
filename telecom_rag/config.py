"""集中管理配置。所有配置都来自环境变量（也可以写在项目根目录的 .env 文件里），代码中不出现任何密钥。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
PLANS_PATH = DATA_DIR / "plans.json"
USERDICT_PATH = DATA_DIR / "userdict.txt"
STORAGE_DIR = ROOT / "storage"
EVAL_PATH = ROOT / "eval" / "eval_set.jsonl"
RESULTS_DIR = ROOT / "results"

# 各后端的默认模型，都可以用 LLM_MODEL 覆盖
DEFAULT_MODELS = {
    "openai": "deepseek-chat",
    "anthropic": "claude-opus-5-5",
    "hf": "Qwen/Qwen3-4B-Instruct-2507",
    "offline": "rule-based",
}
DEFAULT_BASE_URLS = {
    "openai": "https://api.deepseek.com",
    "anthropic": "https://api.anthropic.com",
}


def load_dotenv_if_present() -> None:
    """读取 .env；已经存在的环境变量优先，不会被覆盖。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


@dataclass
class Settings:
    llm_provider: str
    llm_model: str
    llm_base_url: str | None
    llm_api_key: str | None = field(repr=False)  # repr=False：打印配置时不泄露密钥
    llm_effort: str | None
    embed_model: str
    rerank_model: str | None
    retrieval_device: str | None  # 向量模型和重排模型放在哪：None=自动，cpu，cuda
    top_k: int
    candidates: int
    min_rerank_score: float
    chunk_size: int
    max_tool_steps: int
    history_turns: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_if_present()
        provider = (_env("LLM_PROVIDER") or "").lower()
        if not provider:
            # 没有显式指定时自动判断：有密钥就用对应后端，否则退回离线基线，保证一条命令能跑通
            if _env("LLM_API_KEY") or _env("OPENAI_API_KEY"):
                provider = "openai"
            elif _env("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            else:
                provider = "offline"
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"未知的 LLM_PROVIDER={provider}，可选：{', '.join(DEFAULT_MODELS)}")

        api_key = _env("LLM_API_KEY")
        if provider == "openai":
            api_key = api_key or _env("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = api_key or _env("ANTHROPIC_API_KEY")

        rerank = _env("RERANK_MODEL", "BAAI/bge-reranker-base")
        # 本地大模型（hf）要占约 9GB 显存，这时把小的检索模型放到 CPU，避免显存溢出到内存后推理变得极慢
        retrieval_device = _env("RETRIEVAL_DEVICE") or ("cpu" if provider == "hf" else None)
        return cls(
            llm_provider=provider,
            llm_model=_env("LLM_MODEL", DEFAULT_MODELS[provider]),
            # 显式给出默认地址，避免意外继承 shell 里为其他工具设置的 ANTHROPIC_BASE_URL 等变量
            llm_base_url=_env("LLM_BASE_URL", DEFAULT_BASE_URLS.get(provider)),
            llm_api_key=api_key,
            llm_effort=_env("LLM_EFFORT"),
            embed_model=_env("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
            rerank_model=None if rerank.lower() in {"none", "off", "0"} else rerank,
            retrieval_device=retrieval_device,
            top_k=int(_env("TOP_K", "4")),
            candidates=int(_env("CANDIDATES", "20")),
            min_rerank_score=float(_env("MIN_RERANK_SCORE", "0.05")),
            chunk_size=int(_env("CHUNK_SIZE", "350")),
            max_tool_steps=int(_env("MAX_TOOL_STEPS", "4")),
            history_turns=int(_env("HISTORY_TURNS", "6")),
        )

    def describe(self) -> str:
        return f"{self.llm_provider}:{self.llm_model}"
