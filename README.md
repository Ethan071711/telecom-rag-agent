# 运营商客服知识库与工具调用助手

一个可以本机演示的个人学习项目：从虚构运营商资料中检索规则，调用业务函数计算费用、查询和推荐套餐，并展示回答依据。

知识库中的青禾通信、资费与规则为合成示例，不代表真实运营商。项目未在真实企业生产环境部署，代码与说明使用了 AI 辅助工具。

## 先看演示

| 问题 | 主要能力 |
|---|---|
| 携号转网对在网时长有什么要求？ | 检索业务文档，回答附原文引用 |
| 39元套餐用了70GB流量、100分钟电话，要交多少钱？ | 工具提取参数并确定性计算费用 |
| 青春校园卡多少钱？那毕业以后呢？ | 多轮上下文补全，展示追问处理与已知局限 |

[演示顺序与代码讲解](docs/演示与讲解.md) · [详细实现与历史实验](TECHNICAL_NOTES.md) · [本轮整理记录](docs/整理记录_20261024.md)

## Windows 快速运行

Python 3.10—3.13；现有环境在 3.13 上验证。新环境先安装依赖：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:LLM_PROVIDER = 'offline'
.\.venv\Scripts\python.exe run.py web
```

终端会给出本机访问地址。`offline` 是规则路由与原文抽取，不调用付费大模型；向量模型和重排模型首次仍需下载，已有缓存时可断网运行。模型后端的切换见 `.env.example` 和详细说明。

其他入口：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run.py ask "停机保号每个月多少钱？"
.\.venv\Scripts\python.exe run.py eval
```

运行问答与评测前保持 `LLM_PROVIDER=offline`，可避免自动选中环境中已有的云端模型密钥。

## 一个问题怎样得到答案

用户问题 → 标题分块资料 → 向量与 BM25 检索 → RRF 合并和重排 → 规则或大模型选择工具 → 返回回答、原文及工具参数。

业务工具包括 `query_plans`、`calculate_fee`、`recommend_plan`、`search_knowledge`。费用规则由 Python 函数执行；界面展示引用来源，方便人工检查。

## 验证与限制

- 本轮现有 38 项单元测试通过，检查工具、引用、对话处理和网页处理逻辑，不调用付费模型。
- 49 条业务自测与知识库均为项目内合成资料；本轮离线规则基线答对 39/49（79.6%），知识库外拒答 3/6，多轮题答对 1/4。工具题为 8/8；这些是小样本自测结果。完整复现记录见 [2026年10月24日验证](docs/verification/20261024/verification.json)。
- 原有 Qwen、检索消融等结果保留在详细说明和 `results/`，没有将其标为本轮重新运行。
- 规则基线对复杂追问和知识库外问题仍可能答错；本机演示以单队列处理请求，不代表多用户生产服务。
- 自测命中率不能外推为真实客户准确率，也没有人工平均耗时、业务提效或线上 MTTR 的测量。

## 读代码

| 文件 | 作用 |
|---|---|
| `run.py` | 运行入口 |
| `telecom_rag/documents.py` | 文档解析与切块 |
| `telecom_rag/retriever.py` | 检索、融合、重排 |
| `telecom_rag/tools.py` | 业务工具与计算 |
| `telecom_rag/agent.py` | 对话、工具循环、引用映射 |
| `telecom_rag/web.py` | Gradio 演示界面 |
| `tests/` 与 `eval/` | 程序测试和业务自测 |

数据来源说明见 [data/SOURCES.md](data/SOURCES.md)；依赖及其版本见 `requirements.txt`。
