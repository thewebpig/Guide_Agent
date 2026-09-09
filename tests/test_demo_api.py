from fastapi.testclient import TestClient

from guide_agent.agent_contracts import AgentResult
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
