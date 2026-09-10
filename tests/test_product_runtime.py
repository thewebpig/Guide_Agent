import asyncio

import pytest
from fastapi.testclient import TestClient

import guide_agent.app_settings as app_settings_module
from guide_agent.agent_contracts import AgentResult
from guide_agent.answering import ChatService
from guide_agent.capacity import CapacityExceeded, ClientRateLimiter, RequestGate
from guide_agent.demo_api import create_demo_app
from guide_agent.sessions import InMemorySessionStore


class ContextAgent:
    def __init__(self) -> None:
        self.calls = []

    async def arun(self, message, *, history=(), current_location=None):
        self.calls.append((message, history, current_location))
        return AgentResult("completed", f"收到：{message}", (), None, "response-1")

    async def aclose(self):
        pass


def test_http_session_retains_history_location_and_can_be_cleared() -> None:
    agent = ContextAgent()
    app = create_demo_app(service_factory=lambda: ChatService(agent))
    with TestClient(app) as client:
        first = client.post(
            "/chat",
            json={"question": "我在哪里？", "session_id": "session123", "current_location": "lobby"},
        )
        second = client.post(
            "/chat",
            json={"question": "从这里去306怎么走？", "session_id": "session123"},
        )
        cleared = client.delete("/sessions/session123")
        third = client.post(
            "/chat",
            json={"question": "重新开始", "session_id": "session123"},
        )

    assert first.status_code == second.status_code == third.status_code == 200
    assert second.json()["current_location"] == "lobby"
    assert agent.calls[1][1] == (
        {"role": "user", "content": "我在哪里？"},
        {"role": "assistant", "content": "收到：我在哪里？"},
    )
    assert agent.calls[1][2] == "lobby"
    assert cleared.status_code == 204
    assert agent.calls[2][1] == ()
    assert agent.calls[2][2] is None


def test_http_rejects_unverified_or_non_spatial_current_location() -> None:
    app = create_demo_app(service_factory=lambda: ChatService(ContextAgent()))
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "question": "从这里出发",
                "session_id": "session123",
                "current_location": "yang_shanlin_office",
            },
        )
    assert response.status_code == 400
    assert response.json()["status"] == "invalid_location"


def test_http_rejects_unknown_explicit_route_origin_without_calling_model() -> None:
    agent = ContextAgent()
    app = create_demo_app(service_factory=lambda: ChatService(agent))
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "question": "从食堂去王刚教授办公室怎么走？",
                "session_id": "session123",
                "current_location": "main_entrance",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "invalid_start_location"
    assert "食堂" in response.json()["answer"]
    assert "主入口" not in response.json()["answer"]
    assert agent.calls == []


def test_http_accepts_known_alias_as_explicit_route_origin() -> None:
    agent = ContextAgent()
    app = create_demo_app(service_factory=lambda: ChatService(agent))
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "question": "我在一号报告厅，帮我去找任老师。",
                "session_id": "session123",
                "current_location": "main_entrance",
            },
        )

    assert response.status_code == 200
    assert agent.calls[0][0] == "我在一号报告厅，帮我去找任老师。"


def test_readiness_fails_closed_without_server_secret(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        app_settings_module,
        "LOCAL_CONFIG_PATH",
        tmp_path / "missing-config.local.yaml",
    )
    app = create_demo_app(service_factory=lambda: ChatService(ContextAgent()))
    with TestClient(app) as client:
        liveness = client.get("/health")
        readiness = client.get("/ready")
    assert liveness.status_code == 200
    assert readiness.status_code == 503
    assert readiness.json()["ready"] is False


@pytest.mark.anyio
async def test_request_gate_rejects_when_active_capacity_has_no_queue() -> None:
    gate = RequestGate(1, 0)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def occupy() -> None:
        async with gate.slot():
            entered.set()
            await release.wait()

    task = asyncio.create_task(occupy())
    await entered.wait()
    with pytest.raises(CapacityExceeded):
        async with gate.slot():
            pass
    release.set()
    await task


@pytest.mark.anyio
async def test_client_rate_limiter_has_a_fixed_per_client_budget() -> None:
    limiter = ClientRateLimiter(2)
    assert await limiter.allow("client-a")
    assert await limiter.allow("client-a")
    assert not await limiter.allow("client-a")
    assert await limiter.allow("client-b")


@pytest.mark.anyio
async def test_session_history_is_bounded_without_time_expiry() -> None:
    store = InMemorySessionStore(max_turns=2)
    for number in range(3):
        await store.append("session", f"q{number}", f"a{number}")
    history, _ = await store.context("session")
    assert history == (
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    )
