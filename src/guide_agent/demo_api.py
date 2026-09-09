"""FastAPI demo and JSON API for the LangChain + MCP guide agent."""

import asyncio
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from guide_agent.agent_contracts import AgentResult, ToolTrace
from guide_agent.answering import ChatService, TrustedAnswer
from guide_agent.runtime import (
    DEFAULT_SCENE_PATH,
    RuntimeBuildError,
    build_default_chat_service,
)
from guide_agent.runtime_lifecycle import LazyServiceCache
from guide_agent.scene import load_scene

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")
PRESETS = [
    "体验中心常规开放时间是什么？",
    "接待机器人在哪里？",
    "从主入口到人工智能实验室怎么走？",
    "这里可以停车吗？",
]


class DemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class DemoSource(BaseModel):
    source: str
    chunk_id: str
    score: float


class DemoTrace(BaseModel):
    tool_name: str
    arguments: dict[str, object]
    result: object
    call_status: str
    business_status: str | None = None
    # Kept temporarily for clients written before the status split. It is the
    # MCP envelope status, identical to ``call_status``.
    status: str


class DemoResponse(BaseModel):
    request_id: str
    status: str
    answer: str
    tools: list[str]
    sources: list[DemoSource]
    traces: list[DemoTrace]
    architecture: str = "langchain + mcp"
    api_format: str
    elapsed_ms: float
    error: str | None = None


def _api_format() -> str:
    value = os.environ.get("OPENAI_API_FORMAT", "responses").strip() or "responses"
    return value if value in {"responses", "chat_completions"} else "invalid"


def _model_configured() -> bool:
    return bool(
        os.environ.get("OPENAI_MODEL", "").strip()
        and os.environ.get("OPENAI_API_KEY", "").strip()
    )


def _trace(trace: ToolTrace) -> DemoTrace:
    result = {"status": trace.status}
    if isinstance(trace.result, dict) and "data" in trace.result:
        result["data"] = trace.result["data"]
    business_status = trace.business_status
    if business_status is None and isinstance(result.get("data"), dict):
        nested_status = result["data"].get("status")
        business_status = nested_status if isinstance(nested_status, str) else None
    return DemoTrace(
        tool_name=trace.tool_name,
        arguments=trace.arguments,
        result=result,
        call_status=trace.status,
        business_status=business_status,
        status=trace.status,
    )


def _payload(
    request_id: str, answer: TrustedAnswer, result: AgentResult | None, elapsed: float
) -> DemoResponse:
    return DemoResponse(
        request_id=request_id,
        status=answer.status,
        answer=answer.answer,
        tools=list(answer.tools),
        sources=[
            DemoSource(source=item.source, chunk_id=item.chunk_id, score=item.score)
            for item in answer.sources
        ],
        traces=[_trace(item) for item in result.tool_traces] if result else [],
        api_format=_api_format(),
        elapsed_ms=round(elapsed, 2),
        error=answer.error,
    )


def _error(request_id: str, status: str, answer: str, elapsed: float) -> DemoResponse:
    return DemoResponse(
        request_id=request_id,
        status=status,
        answer=answer,
        tools=[],
        sources=[],
        traces=[],
        api_format=_api_format(),
        elapsed_ms=round(elapsed, 2),
        error=status,
    )


def create_demo_app(
    *,
    service_factory: Callable[[], ChatService] | None = None,
    timeout_seconds: float = 90.0,
) -> FastAPI:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    cache = LazyServiceCache(
        lambda _: (
            service_factory() if service_factory else build_default_chat_service()
        ),
        close_owned=True,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await cache.aclose()

    app = FastAPI(title="Guide Agent", version="1.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "demo.html")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "architecture": "langchain + mcp",
            "api_format": _api_format(),
            "model_configured": _model_configured(),
        }

    @app.get("/demo/config")
    def config() -> dict[str, object]:
        scene = load_scene(DEFAULT_SCENE_PATH)
        return {
            "architecture": "langchain + mcp",
            "api_format": _api_format(),
            "model_configured": _model_configured(),
            "scene": {"name": scene.name, "preset_questions": PRESETS},
        }

    @app.post("/chat", response_model=DemoResponse)
    @app.post("/demo/chat", response_model=DemoResponse)
    async def chat(request: DemoRequest, response: Response) -> DemoResponse:
        request_id = uuid4().hex
        started = perf_counter()
        try:
            service = await cache.get("default")
            answer, result = await asyncio.wait_for(
                service.aanswer_with_trace(request.question), timeout=timeout_seconds
            )
            payload = _payload(
                request_id, answer, result, (perf_counter() - started) * 1000
            )
        except TimeoutError:
            response.status_code = 504
            payload = _error(
                request_id,
                "timeout",
                "请求处理超时，请稍后重试。",
                (perf_counter() - started) * 1000,
            )
        except RuntimeBuildError:
            response.status_code = 503
            message = (
                "服务初始化失败，请检查场景和模型配置后重试。"
                if _model_configured()
                else "模型服务尚未配置。请设置 OPENAI_MODEL 和 OPENAI_API_KEY 后重启服务。"
            )
            payload = _error(
                request_id,
                "service_unavailable",
                message,
                (perf_counter() - started) * 1000,
            )
        except Exception:
            LOGGER.exception("request failed request_id=%s", request_id)
            response.status_code = 500
            payload = _error(
                request_id,
                "internal_error",
                "服务内部错误，请稍后重试。",
                (perf_counter() - started) * 1000,
            )
        return payload

    return app


app = create_demo_app()
