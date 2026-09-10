# 合肥工业大学工程管理与智能制造研究中心智能导览

面向访客的本地导览产品：浏览器访问 FastAPI，LangChain Agent 调用部署人员
配置的 OpenAI 兼容模型，地点查询、知识检索和路线规划全部通过本地 MCP stdio
工具服务执行。浏览器不接触模型地址和 API Key。

## 一期范围

- 场景：合肥工业大学屯溪路校区工程管理与智能制造研究中心（管理学院）。
- 数据：24 个地点/人物/知识实体、19 条仿真路线边，以及中心和教师公开资料。
- 能力：场馆知识问答、POI 查询、教师办公室查询、Dijkstra 路线、连续对话、当前位置上下文。
- 安全：内部区域、未核实办公室和非空间人物资料由路线代码强制拒绝，不依赖模型自觉。
- 容量：模型并发、等待队列、单 IP 限流和请求超时均由 `config.yaml` 配置。

路线坐标和距离均为算法验证使用的仿真数据，不是建筑测绘坐标或真实米数；实际使用应以现场标识为准。

## 架构

```text
浏览器
  ↓ HTTP
FastAPI（会话、限流、超时、日志）
  ↓
LangChain Agent → OpenAI 兼容模型（服务端配置）
  ↓ tool call
MCP stdio 子进程
  ├─ search_knowledge → FastEmbed + FAISS
  ├─ lookup_poi
  └─ plan_route → 权限检查 + Dijkstra
```

FastAPI 首次真正需要工具时启动同机 MCP 子进程。MCP 不开放公网端口，也不加载模型 API Key。

## 配置

非密钥配置统一位于 [`config.yaml`](config.yaml)。模型可以替换，只要服务提供
OpenAI 兼容接口并支持本项目需要的 Tool Calling。下面仅展示配置结构，尖括号
内容是占位符，不能原样使用：

```yaml
model:
  name: <服务商返回的模型 ID>
  base_url: <OpenAI 兼容 API 根地址>
  api_format: <chat_completions 或 responses>

scene:
  path: <场景 JSON 路径>

retrieval:
  minimum_score: <按检索评测确定的阈值>
```

不要根据控制台卡片标题猜测模型名，应优先使用服务商模型列表接口或调用示例中的
精确模型 ID。更换模型后，应重新运行端到端 Tool Calling、路线和超时测试。

模型相关环境变量会覆盖 YAML 中的对应配置：

```text
OPENAI_MODEL
OPENAI_BASE_URL
OPENAI_API_FORMAT
OPENAI_API_KEY
```

API Key 支持三种配置方式，优先级为：启动参数/环境变量 > `config.local.yaml` > `config.yaml`。

推荐复制私有配置模板：

```powershell
Copy-Item config.local.example.yaml config.local.yaml
```

然后只修改不会被 Git 提交的 `config.local.yaml`：

```yaml
model:
  api_key: 实际密钥
```

也可以临时使用服务器环境变量：

```powershell
$env:OPENAI_API_KEY = "实际密钥"
```

程序也允许在 `config.yaml` 的 `model.api_key` 中配置，但该文件受 Git 跟踪，
不推荐这样做。不要把真实密钥提交到 GitHub。

## 本地启动

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync --frozen
.\scripts\start_demo.ps1 -ConfigureSecret
```

也可以通过启动参数覆盖配置文件：

```powershell
.\scripts\start_demo.ps1 -ApiKey "临时Key"
```

命令行参数可能进入终端历史，因此日常使用仍推荐 `config.local.yaml` 或交互式
`-ConfigureSecret`。启动后游客访问 `server.host` 和 `server.port` 对应的地址；
仓库当前值请直接查看 [`config.yaml`](config.yaml)，不要依赖 README 中的固定端口。

页面不会要求游客填写模型配置。`GET /health` 可用于存活检查，Swagger UI 位于 `/docs`。

## Docker 本地运行

安装 Docker Desktop 后：

```powershell
$env:OPENAI_API_KEY = "实际密钥"
docker compose up --build
```

镜像启动一个 Uvicorn/FastAPI 进程；FastAPI 再按需启动同一容器内的 MCP 子进程。
镜像构建阶段预下载本地中文 Embedding 模型，避免首位游客触发下载。

## 会话行为

- 浏览器自动创建随机会话 ID，服务端按 `session.max_turns` 保存最近若干轮上下文。
- 会话没有主动时间过期；点击“新对话”会立即清除。
- 本地版本使用进程内存，重启服务后会话清空。
- 当前位置由页面明确选择，并由服务器校验为可导航 POI。

## 验证

```powershell
uv run pytest -q
uv run python scripts/evaluate_retrieval.py
```

测试不调用大模型 API。测试集覆盖原有 LangChain/MCP 契约、真实场景解析、知识文档、
23 条指定路线、限制区域拒绝、未核实办公室拒绝、配置、会话和 HTTP 接口。

真实模型端到端测试必须由部署人员在本机注入有效 API Key 后单独执行；更换模型时
必须重新验证 Tool Calling、回答质量和延迟，测试结果及密钥不得提交到仓库。

内容更新、导航状态、健康检查和故障处理见 [`docs/local_operations.md`](docs/local_operations.md)。

## 目录

```text
config.yaml                         服务器非密钥配置
scenes/hfut_management_center/      一期真实场景与知识文档
src/guide_agent/                    Agent、MCP、业务层和 FastAPI
tests/                              离线自动化测试
Dockerfile / compose.yaml           本地容器运行
docs/hfut_route_acceptance.md       用户提供的路线验收基准
```

项目采用 [MIT License](LICENSE)。
