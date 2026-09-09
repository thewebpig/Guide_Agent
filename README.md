# Guide Agent

一个可直接演示的导览 Agent：FastAPI 页面/API → LangChain Agent → OpenAI-compatible 模型服务 → MCP stdio 工具服务。业务工具只有三个：知识检索、地点查询、路线规划。

模型负责选择工具，检索、POI 和最短路线由受约束的 Python 工具执行，回答中的引用与路线来自真实工具轨迹。

## 架构

```text
Browser / HTTP client
        ↓
FastAPI → LangChain create_agent → Responses 或 Chat Completions
                                        ↓ tool call
                              MCP client → MCP stdio server
                                           ├─ search_knowledge
                                           ├─ lookup_poi
                                           └─ plan_route
```

## 快速开始

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync --frozen
.\scripts\start_demo.ps1 -ConfigureModel -ApiFormat chat_completions
```

按提示输入模型、API 根地址和 Key，然后打开 `http://127.0.0.1:8765`。

`OPENAI_BASE_URL` 填服务商的 API 根地址，例如 `https://provider.example/v1`，不要填 `/chat/completions` 或 `/responses`。若服务商支持 Responses API，把 `-ApiFormat` 改为 `responses`。

| 配置值 | 实际端点 |
|---|---|
| `responses` | `{OPENAI_BASE_URL}/responses` |
| `chat_completions` | `{OPENAI_BASE_URL}/chat/completions` |

也可手动配置环境变量：

```powershell
$env:OPENAI_MODEL = "MiniCPM5-2B"
$env:OPENAI_BASE_URL = "https://provider.example/v1"
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_API_FORMAT = "chat_completions"
uv run uvicorn guide_agent.demo_api:app --host 127.0.0.1 --port 8765
```

## HTTP API

`GET /health` 只报告架构、配置状态和 API 格式，不返回密钥。

`POST /chat`：

```json
{"question":"从主入口到人工智能实验室怎么走？"}
```

响应包含 `answer`、`sources`、`traces`、`api_format` 与 `elapsed_ms`。Swagger UI 位于 `/docs`，完整设计与演示话术见 [架构说明](docs/architecture.md) 和 [三分钟演示](docs/demo_3_minutes.md)。

## 验证

```powershell
uv run pytest
uv run python -c "from guide_agent.demo_api import app; print(app.title)"
```

测试不调用大模型 API：覆盖场景解析、RAG、POI、Dijkstra 路线、工具 Schema、FastAPI、MCP stdio 工具发现，以及两种模型协议的真实 SDK 请求路径，不消耗 API 额度。首次运行 RAG/MCP 测试时，FastEmbed 可能需要下载 Embedding 模型；缓存后可离线执行。

## 目录

```text
src/guide_agent/   Agent、MCP、业务工具和 FastAPI
scenes/demo/       完全虚构的演示场景与知识文档
tests/             104 个离线测试
scripts/           PowerShell 启动脚本和 CLI
docs/              架构、限制与演示说明
```

这个仓库不会提交 `.env`、密钥、私有求职材料、历史评测输出或模型调用记录。项目采用 [MIT License](LICENSE)。
