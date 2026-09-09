from fastapi.testclient import TestClient

from guide_agent.agent_contracts import AgentResult, ToolTrace
from guide_agent.answering import ChatService
from guide_agent.demo_api import create_demo_app


class FakeAgent:
    async def arun(self, message: str) -> AgentResult:
        return AgentResult("completed", f"收到：{message}", (), None, "response-1")

    async def aclose(self) -> None: pass


def test_health_and_chat_expose_fixed_architecture(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "demo")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("OPENAI_API_FORMAT", "chat_completions")
    app = create_demo_app(service_factory=lambda: ChatService(FakeAgent()))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.json() == {"ok": True, "architecture": "langchain + mcp", "api_format": "chat_completions", "model_configured": True}
        response = client.post("/chat", json={"question": "你好"})
    assert response.status_code == 200
    body = response.json()
    assert body["architecture"] == "langchain + mcp"
    assert body["answer"] == "收到：你好"


def test_chat_rejects_legacy_backend_selection() -> None:
    app = create_demo_app(service_factory=lambda: ChatService(FakeAgent()))
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "你好", "backend": "native"})
    assert response.status_code == 422


class BusinessFailureTraceAgent:
    async def arun(self, message: str) -> AgentResult:
        trace = ToolTrace(
            tool_name="plan_route",
            arguments={"start_id": "entrance:主入口[入口,大门]", "end_id": "ai_lab"},
            result={
                "status": "ok",
                "tool_name": "plan_route",
                "data": {
                    "status": "invalid_input",
                    "path": [],
                    "distance": None,
                },
            },
            status="ok",
        )
        return AgentResult(
            "completed",
            "路线：entrance -> ai_lab；总距离：14。",
            (trace,),
            None,
            "response-1",
        )

    async def aclose(self) -> None: pass


def test_chat_trace_separates_mcp_call_from_business_result() -> None:
    app = create_demo_app(
        service_factory=lambda: ChatService(BusinessFailureTraceAgent())
    )
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "从主入口到实验室怎么走"})

    assert response.status_code == 200
    trace = response.json()["traces"][0]
    assert trace["call_status"] == "ok"
    assert trace["business_status"] == "invalid_input"
    assert trace["status"] == "ok"  # Backward-compatible envelope status.
    assert trace["result"]["data"]["status"] == "invalid_input"


class KnowledgeTraceAgent:
    async def arun(self, message: str) -> AgentResult:
        return AgentResult(
            "completed",
            "ignored by trusted-answer layer",
            (
                ToolTrace(
                    tool_name="search_knowledge",
                    arguments={"query": "体验中心常规开放时间是什么？", "top_k": 5},
                    result={
                        "status": "ok",
                        "tool_name": "search_knowledge",
                        "data": [
                            {
                                "source": "visitor_guide.md",
                                "chunk_id": "opening-hours",
                                "text": "This full document text must not enter the trace.",
                                "score": 0.499150,
                            }
                        ],
                        "error": None,
                    },
                    status="ok",
                ),
            ),
            None,
            "response-1",
        )

    async def aclose(self) -> None: pass


def test_chat_exposes_sanitized_retrieval_observability() -> None:
    app = create_demo_app(service_factory=lambda: ChatService(KnowledgeTraceAgent()))
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "开放时间是什么？"})

    assert response.status_code == 200
    trace = response.json()["traces"][0]
    assert trace["retrieval"] == {
        "top_score": 0.49915,
        "minimum_score": 0.50,
        "accepted": False,
    }
    assert trace["result"]["data"] == [
        {"source": "visitor_guide.md", "chunk_id": "opening-hours", "score": 0.49915}
    ]
    assert "This full document text" not in str(trace)
