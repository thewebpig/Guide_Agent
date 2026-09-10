"""FastAPI demo and JSON API for the LangChain + MCP guide agent."""

import asyncio
import logging
import os
import re
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from guide_agent.agent_contracts import AgentResult, ToolTrace
from guide_agent.answering import ChatService, TrustedAnswer
from guide_agent.app_settings import AppSettings, load_app_settings
from guide_agent.capacity import CapacityExceeded, ClientRateLimiter, RequestGate
from guide_agent.runtime import (
    RuntimeBuildError,
    build_default_chat_service,
    resolve_scene_path,
)
from guide_agent.runtime_lifecycle import LazyServiceCache
from guide_agent.scene import load_scene
from guide_agent.sessions import InMemorySessionStore

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")
PRESETS = [
    "工程管理与智能制造研究中心位于哪个校区？",
    "从主入口到第二学术报告厅怎么走？",
    "从主入口带我去王刚老师办公室。",
    "杨善林院士的办公室在哪里？",
]


def _normalized_location_label(value: str) -> str:
    normalized = re.sub(r"[\s，,。！？!?（）()]", "", value)
    return normalized.replace("老师", "").replace("教授", "")


def _explicit_route_origin(question: str) -> str | None:
    """Extract an origin that the visitor explicitly stated in this turn."""

    patterns = (
        r"我(?:现在)?在(.+?)[，,]",
        r"从(.+?)(?:怎么)?(?:带我)?(?:到|去|前往)",
    )
    for pattern in patterns:
        match = re.search(pattern, question)
        if not match:
            continue
        origin = match.group(1).strip(" ，,。！？!?")
        if origin and origin not in {"这里", "当前位置", "这儿"}:
            return origin
    return None


def _has_navigable_location(origin: str, pois: list[object]) -> bool:
    expected = _normalized_location_label(origin)
    for poi in pois:
        if getattr(poi, "entity_type", None) != "place":
            continue
        if getattr(poi, "navigation_status", None) != "navigable":
            continue
        labels = [getattr(poi, "name", ""), *getattr(poi, "aliases", [])]
        if any(_normalized_location_label(label) == expected for label in labels):
            return True
    return False


class DemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    current_location: str | None = Field(default=None, min_length=1, max_length=100)

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


class DemoRetrieval(BaseModel):
    """Retrieval observability without returning retrieved document text."""

    top_score: float | None
    minimum_score: float
    accepted: bool


class DemoTrace(BaseModel):
    tool_name: str
    arguments: dict[str, object]
    result: object
    call_status: str
    business_status: str | None = None
    # Kept temporarily for clients written before the status split. It is the
    # MCP envelope status, identical to ``call_status``.
    status: str
    query_normalized: bool = False
    retrieval: DemoRetrieval | None = None


class DemoResponse(BaseModel):
    request_id: str
    session_id: str
    current_location: str | None
    status: str
    answer: str
    tools: list[str]
    sources: list[DemoSource]
    traces: list[DemoTrace]
    architecture: str = "langchain + mcp"
    api_format: str
    elapsed_ms: float
    error: str | None = None


def _api_format(settings: AppSettings | None = None) -> str:
    default = settings.model.api_format if settings else "responses"
    value = os.environ.get("OPENAI_API_FORMAT", default).strip() or default
    return value if value in {"responses", "chat_completions"} else "invalid"


def _model_configured(settings: AppSettings | None = None) -> bool:
    from_environment = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    from_file = bool(settings and settings.model.api_key)
    return from_environment or from_file


def _retrieval_observability(
    trace: ToolTrace, minimum_score: float
) -> DemoRetrieval | None:
    if trace.tool_name != "search_knowledge":
        return None
    payload = trace.result.get("data") if isinstance(trace.result, dict) else None
    scores = (
        [
            float(item["score"])
            for item in payload
            if isinstance(item, dict)
            and isinstance(item.get("score"), (int, float))
            and not isinstance(item.get("score"), bool)
        ]
        if isinstance(payload, list)
        else []
    )
    top_score = max(scores, default=None)
    return DemoRetrieval(
        top_score=top_score,
        minimum_score=minimum_score,
        accepted=top_score is not None and top_score >= minimum_score,
    )


def _trace(trace: ToolTrace, *, minimum_score: float) -> DemoTrace:
    result = {"status": trace.status}
    if isinstance(trace.result, dict) and "data" in trace.result:
        data = trace.result["data"]
        # Search results may contain full document chunks.  The response has
        # separately audited sources, so the trace only needs IDs and scores.
        if trace.tool_name == "search_knowledge" and isinstance(data, list):
            result["data"] = [
                {
                    key: item[key]
                    for key in ("source", "chunk_id", "score")
                    if key in item
                }
                for item in data
                if isinstance(item, dict)
            ]
        else:
            result["data"] = data
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
        query_normalized=trace.query_normalized,
        retrieval=_retrieval_observability(trace, minimum_score),
    )


def _payload(
    request_id: str,
    session_id: str,
    current_location: str | None,
    answer: TrustedAnswer,
    result: AgentResult | None,
    elapsed: float,
    *,
    minimum_score: float,
    api_format: str,
) -> DemoResponse:
    return DemoResponse(
        request_id=request_id,
        session_id=session_id,
        current_location=current_location,
        status=answer.status,
        answer=answer.answer,
        tools=list(answer.tools),
        sources=[
            DemoSource(source=item.source, chunk_id=item.chunk_id, score=item.score)
            for item in answer.sources
        ],
        traces=[_trace(item, minimum_score=minimum_score) for item in result.tool_traces]
        if result
        else [],
        api_format=api_format,
        elapsed_ms=round(elapsed, 2),
        error=answer.error,
    )


def _error(
    request_id: str,
    session_id: str,
    current_location: str | None,
    status: str,
    answer: str,
    elapsed: float,
    *,
    api_format: str,
) -> DemoResponse:
    return DemoResponse(
        request_id=request_id,
        session_id=session_id,
        current_location=current_location,
        status=status,
        answer=answer,
        tools=[],
        sources=[],
        traces=[],
        api_format=api_format,
        elapsed_ms=round(elapsed, 2),
        error=status,
    )


def create_demo_app(
    *,
    service_factory: Callable[[], ChatService] | None = None,
    timeout_seconds: float | None = None,
) -> FastAPI:
    product = load_app_settings()
    request_timeout = (
        product.server.request_timeout_seconds
        if timeout_seconds is None
        else timeout_seconds
    )
    if request_timeout <= 0:
        raise ValueError("timeout_seconds must be greater than 0")
    scene_path = resolve_scene_path(settings=product)
    scene = load_scene(scene_path)
    pois_by_id = {poi.id: poi for poi in scene.pois}
    gate = RequestGate(product.server.model_concurrency, product.server.queue_size)
    rate_limiter = ClientRateLimiter(product.server.requests_per_minute)
    sessions = InMemorySessionStore(product.session.max_turns)
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

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        started = perf_counter()
        request_id = request.headers.get("x-request-id", uuid4().hex)
        try:
            result = await call_next(request)
        except Exception:
            LOGGER.exception("http_request_failed request_id=%s path=%s", request_id, request.url.path)
            raise
        result.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "http_request request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            result.status_code,
            (perf_counter() - started) * 1000,
        )
        return result

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "demo.html")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "architecture": "langchain + mcp",
            "api_format": _api_format(product),
            "model_configured": _model_configured(product),
        }

    @app.get("/ready")
    def ready(response: Response) -> dict[str, object]:
        configured = _model_configured(product)
        if not configured:
            response.status_code = 503
        return {
            "ready": configured,
            "scene_loaded": True,
            "model_configured": configured,
        }

    @app.get("/demo/config")
    def config() -> dict[str, object]:
        return {
            "architecture": "langchain + mcp",
            "api_format": _api_format(product),
            "model_configured": _model_configured(product),
            "scene": {
                "name": scene.name,
                "preset_questions": PRESETS,
                "default_location": "main_entrance",
                "locations": [
                    {"id": poi.id, "name": poi.name, "floor": poi.position.floor if poi.position else None}
                    for poi in scene.pois
                    if poi.entity_type == "place" and poi.navigation_status == "navigable"
                ],
                "route_notice": "路线坐标及权重为仿真数据，不是测绘坐标或真实米数。具体走法请以现场标识为准。",
            },
        }

    @app.delete("/sessions/{session_id}", status_code=204)
    async def clear_session(session_id: str) -> Response:
        await sessions.clear(session_id)
        return Response(status_code=204)

    @app.post("/chat", response_model=DemoResponse)
    @app.post("/demo/chat", response_model=DemoResponse)
    async def chat(
        request: DemoRequest,
        response: Response,
        http_request: Request,
    ) -> DemoResponse:
        request_id = uuid4().hex
        session_id = request.session_id or uuid4().hex
        started = perf_counter()
        current_location = request.current_location
        try:
            client_id = http_request.client.host if http_request.client else "unknown"
            if not await rate_limiter.allow(client_id):
                response.status_code = 429
                response.headers["Retry-After"] = "60"
                return _error(
                    request_id,
                    session_id,
                    current_location,
                    "rate_limited",
                    "请求过于频繁，请稍后再试。",
                    (perf_counter() - started) * 1000,
                    api_format=_api_format(product),
                )
            if current_location is not None:
                poi = pois_by_id.get(current_location)
                if poi is None or poi.entity_type != "place" or poi.navigation_status != "navigable":
                    response.status_code = 400
                    return _error(
                        request_id,
                        session_id,
                        None,
                        "invalid_location",
                        "当前位置不是可用的导航地点，请重新选择。",
                        (perf_counter() - started) * 1000,
                        api_format=_api_format(product),
                    )
            explicit_origin = _explicit_route_origin(request.question)
            if explicit_origin and not _has_navigable_location(explicit_origin, scene.pois):
                return _error(
                    request_id,
                    session_id,
                    current_location,
                    "invalid_start_location",
                    f"当前场景中没有找到“{explicit_origin}”对应的可导航起点，请重新选择或说明起点。",
                    (perf_counter() - started) * 1000,
                    api_format=_api_format(product),
                )
            history, current_location = await sessions.context(session_id, current_location)
            async with asyncio.timeout(request_timeout):
                async with gate.slot():
                    service = await cache.get("default")
                    if history or current_location:
                        answer, result = await service.aanswer_with_context(
                            request.question,
                            history=history,
                            current_location=current_location,
                        )
                    else:
                        answer, result = await service.aanswer_with_trace(request.question)
            await sessions.append(session_id, request.question, answer.answer)
            payload = _payload(
                request_id,
                session_id,
                current_location,
                answer,
                result,
                (perf_counter() - started) * 1000,
                minimum_score=service.minimum_score,
                api_format=_api_format(product),
            )
        except CapacityExceeded:
            response.status_code = 429
            response.headers["Retry-After"] = "5"
            payload = _error(
                request_id,
                session_id,
                current_location,
                "over_capacity",
                "当前访问人数较多，请稍后重试。",
                (perf_counter() - started) * 1000,
                api_format=_api_format(product),
            )
        except TimeoutError:
            response.status_code = 504
            payload = _error(
                request_id,
                session_id,
                current_location,
                "timeout",
                "请求处理超时，请稍后重试。",
                (perf_counter() - started) * 1000,
                api_format=_api_format(product),
            )
        except RuntimeBuildError:
            response.status_code = 503
            message = (
                "服务初始化失败，请检查场景和模型配置后重试。"
                if _model_configured(product)
                else "模型服务尚未配置。请由部署人员设置 OPENAI_API_KEY 后重启服务。"
            )
            payload = _error(
                request_id,
                session_id,
                current_location,
                "service_unavailable",
                message,
                (perf_counter() - started) * 1000,
                api_format=_api_format(product),
            )
        except Exception:
            LOGGER.exception("request failed request_id=%s", request_id)
            response.status_code = 500
            payload = _error(
                request_id,
                session_id,
                current_location,
                "internal_error",
                "服务内部错误，请稍后重试。",
                (perf_counter() - started) * 1000,
                api_format=_api_format(product),
            )
        return payload

    return app


app = create_demo_app()
