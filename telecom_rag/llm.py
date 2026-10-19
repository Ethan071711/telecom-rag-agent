"""大模型后端。所有后端实现同一个接口 chat(messages, tools) -> Reply，智能体代码与具体厂商解耦。

内部消息格式（接近 OpenAI 风格）：
  {"role": "system" | "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [ToolCall], "raw": 厂商原始返回(可选)}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}

后端：
  openai    兼容 OpenAI 接口的服务：DeepSeek、通义千问(DashScope)、智谱、本地 Ollama 等
  anthropic Claude API（官方 anthropic SDK）
  hf        transformers 直接在本地 GPU/CPU 上加载 Qwen 等模型，手动解析 <tool_call>
  offline   不调用任何大模型的规则基线：正则路由工具 + 抽取式回答，用于无密钥环境和对比实验
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .textutil import REFUSAL_TEXT


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # 厂商原始的 assistant 内容，继续对话时原样回传（Claude 的 thinking 块需要这样做）


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


class OpenAICompatLLM:
    """兼容 OpenAI Chat Completions 接口的服务，覆盖国产大模型 API 和本地 Ollama。"""

    def __init__(self, settings: Settings):
        from openai import OpenAI

        base_url = settings.llm_base_url
        api_key = settings.llm_api_key
        if not api_key and base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
            api_key = "ollama"  # Ollama 不校验密钥，但客户端要求非空
        if not api_key:
            raise ValueError("LLM_PROVIDER=openai 需要设置 LLM_API_KEY（本地 Ollama 除外）")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
        self.model = settings.llm_model

    def chat(self, messages: list[dict], tools: list[dict], context=None) -> Reply:
        payload = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                payload.append({
                    "role": "assistant", "content": m.get("content") or "",
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                                   for c in m["tool_calls"]],
                })
            elif m["role"] == "tool":
                payload.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                payload.append({"role": m["role"], "content": m["content"]})
        kwargs: dict = {"model": self.model, "messages": payload, "temperature": 0}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [ToolCall(c.id, c.function.name, _parse_args(c.function.arguments)) for c in (msg.tool_calls or [])]
        return Reply(content=msg.content or "", tool_calls=calls)


class AnthropicLLM:
    """Claude API。使用官方 anthropic SDK 的 Messages API 手动工具循环。"""

    def __init__(self, settings: Settings):
        import anthropic

        if not settings.llm_api_key:
            raise ValueError("LLM_PROVIDER=anthropic 需要设置 ANTHROPIC_API_KEY 或 LLM_API_KEY")
        self.client = anthropic.Anthropic(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        self.model = settings.llm_model
        self.effort = settings.llm_effort  # 可选：low / medium / high，客服问答用 low 更快更省

    def chat(self, messages: list[dict], tools: list[dict], context=None) -> Reply:
        system, payload = "", []
        for m in messages:
            if m["role"] == "system":
                system += m["content"]
            elif m["role"] == "assistant":
                if m.get("raw") is not None:
                    payload.append({"role": "assistant", "content": m["raw"]})  # 原样回传，包含 thinking 块
                    continue
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                           for c in m.get("tool_calls") or []]
                payload.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                last = payload[-1] if payload else None
                if last and last["role"] == "user" and isinstance(last["content"], list):
                    last["content"].append(block)  # 同一轮的多个工具结果放进同一条 user 消息
                else:
                    payload.append({"role": "user", "content": [block]})
            else:
                payload.append({"role": "user", "content": m["content"]})

        kwargs: dict = {
            "model": self.model, "max_tokens": 8000, "system": system, "messages": payload,
            "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                      for t in tools],
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            return Reply(content=REFUSAL_TEXT)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, _parse_args(b.input)) for b in resp.content if b.type == "tool_use"]
        return Reply(content=text, tool_calls=calls, raw=resp.content if calls else None)


class HFLocalLLM:
    """transformers 本地推理。Qwen 的对话模板原生支持 tools 参数，模型会输出 <tool_call>{...}</tool_call>。"""

    TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

    def __init__(self, settings: Settings, max_new_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
        # 整个模型放在一张 GPU 上。不用 device_map="auto"：显存不够时它会悄悄把部分层放到 CPU，推理变得很慢却不报错
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(settings.llm_model, dtype="auto", device_map=device_map)
        self.model.eval()
        cfg = self.model.generation_config
        cfg.do_sample, cfg.temperature, cfg.top_p, cfg.top_k = False, None, None, None  # 贪心解码，评测结果可复现
        cfg.max_new_tokens = max_new_tokens

    def chat(self, messages: list[dict], tools: list[dict], context=None) -> Reply:
        hf_messages = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                hf_messages.append({"role": "assistant", "content": m.get("content") or "",
                                    "tool_calls": [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                                                   for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                hf_messages.append({"role": "tool", "name": m.get("name"), "content": m["content"]})
            else:
                hf_messages.append({"role": m["role"], "content": m["content"]})
        prompt = self.tokenizer.apply_chat_template(
            hf_messages, tools=[{"type": "function", "function": t} for t in tools] or None,
            add_generation_prompt=True, tokenize=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.model.generation_config)
        text = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if self.torch.cuda.is_available():
            # 每次生成的序列长度不同，PyTorch 缓存的显存会越积越多；12GB 显卡上积到上限后，
            # Windows 驱动会把显存溢出到内存，推理变得极慢。所以每次生成完把缓存还给驱动。
            del output, inputs
            self.torch.cuda.empty_cache()

        calls = []
        for i, block in enumerate(self.TOOL_CALL_RE.findall(text)):
            obj = _parse_args(block)
            if obj.get("name"):
                calls.append(ToolCall(f"call_{i}", obj["name"], _parse_args(obj.get("arguments"))))
        content = self.TOOL_CALL_RE.sub("", text).strip()
        return Reply(content=content, tool_calls=calls)


def create_llm(settings: Settings, plans=None):
    provider = settings.llm_provider
    if provider == "openai":
        return OpenAICompatLLM(settings)
    if provider == "anthropic":
        return AnthropicLLM(settings)
    if provider == "hf":
        return HFLocalLLM(settings)
    if provider == "offline":
        from .offline import OfflineLLM

        return OfflineLLM(plans)
    raise ValueError(f"未知的 LLM_PROVIDER: {provider}")
