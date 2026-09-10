# 本地运行与内容更新

## 首次启动

1. 安装 Python 3.11 和 uv。
2. 执行 `uv sync --frozen`。
3. 复制 `config.local.example.yaml` 为 `config.local.yaml` 并填写 AMD API Key，或执行 `.\scripts\start_demo.ps1 -ConfigureSecret` 交互输入。
4. 浏览器访问 `http://127.0.0.1:8765`。

模型地址、模型名、协议、场景路径、阈值和容量统一由 `config.yaml` 管理。Key 可放在私有 `config.local.yaml`；`-ApiKey` 启动参数或 `OPENAI_API_KEY` 环境变量会覆盖文件值。游客页面没有模型配置入口。

## 更新场景内容

- POI、别名、导航状态和路线：编辑 `scenes/hfut_management_center/scene.json`。
- 场馆知识：编辑 `scenes/hfut_management_center/docs/hfut_management_center_guide.md`。
- 检索验收题：编辑 `evals/hfut_retrieval.json`。
- 路线验收基准：以 `docs/hfut_route_acceptance.md` 为依据更新 `tests/test_hfut_product.py`。

导航状态含义：

- `navigable`：可以作为实体路线起终点。
- `simulation_only`：只参与明确标注的仿真路线。
- `restricted`：内部区域，路线工具强制拒绝作为终点。
- `location_unverified`：具体位置没有公开核实，强制拒绝导航。
- `not_navigable`：人物或知识资料，不是实体地点。

修改后必须执行：

```powershell
uv run pytest -q
uv run python scripts/evaluate_retrieval.py
```

场景及知识索引在进程启动后加载。内容更新通过 Git 审核并重启服务后生效；一期不提供匿名网页后台，避免访客直接篡改正式数据。

## 健康检查与日志

- `/health`：进程存活、架构和配置状态。
- `/ready`：场景已加载且服务器已注入模型密钥时返回 200，否则返回 503。
- HTTP 日志输出请求编号、路径、状态码和耗时，不记录 API Key，也不记录问题正文。
- 错误响应只返回稳定业务文案，详细异常保留在服务器日志中。

## 容量与会话

- 同时执行模型请求：6。
- 等待队列：14；满后返回 HTTP 429。
- 单客户端：每分钟 30 次；超出后返回 HTTP 429。
- 总请求目标超时：15 秒；超时返回 HTTP 504，并清理异常 MCP 会话。
- 会话不设置时间失效，但只保留最近 20 轮；进程重启后清空。
- 本地版本固定一个 Uvicorn worker，保证 MCP/FAISS 只有一份。以后多进程部署必须改成每进程独立资源或外部服务。

## 当前边界

- 坐标和距离是仿真数据，不是实测值。
- 未接入学院平面图、Logo/VIS、门禁、停车、开放时间和无障碍长期规则。
- `/ready` 只检查密钥是否注入，不主动消耗模型额度探测远程服务。
- 告警平台、HTTPS、域名和 Cloudflare 部署留到服务器部署阶段。
