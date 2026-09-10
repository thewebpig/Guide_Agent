"""The project's only agent implementation: LangChain + MCP."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError
from pydantic import ValidationError

from guide_agent.agent_contracts import AgentResult, ToolTrace
from guide_agent.mcp_client import MCPProtocolError, MCPToolClient
from guide_agent.model_settings import ModelSettings
from guide_agent.prompts import DEVELOPER_INSTRUCTIONS
from guide_agent.tools import TOOL_DEFINITIONS


class _MaxStepsExceeded(RuntimeError):
    pass


class _ModelFailure(RuntimeError):
    pass


class _ToolExecutionStopped(RuntimeError):
    pass


def build_chat_model(
    settings: ModelSettings,
    *,
    http_client: Any | None = None,
    http_async_client: Any | None = None,
) -> ChatOpenAI:
    """Create the single provider boundary used by the LangChain agent.

    The injectable transports make endpoint serialization testable without a
    real API key or network access.
    """

    options: dict[str, Any] = {
        "model": settings.model,
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "timeout": 30,
        "max_retries": 0,
        "use_responses_api": settings.api_format == "responses",
        "http_socket_options": (),
    }
    if http_client is not None:
        options["http_client"] = http_client
    if http_async_client is not None:
        options["http_async_client"] = http_async_client
    return ChatOpenAI(**options)


class _MCPMiddleware(AgentMiddleware):
    """Validate MCP calls and retain a compact tool trace."""

    def __init__(
        self,
        client: MCPToolClient,
        traces: list[ToolTrace],
        max_steps: int,
        timeout: float,
        user_message: str,
    ) -> None:
        super().__init__()
        self._client = client
        self._traces = traces
        self._max_steps = max_steps
        self._tool_timeout_seconds = timeout
        self._user_message = user_message.strip()
        self._tool_rounds = 0
        self._lock = asyncio.Lock()
        self.aborted = False
        self._abort_reason = "MCP tool execution failed"

    @hook_config(can_jump_to=["end"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, AIMessage):
                if message.tool_calls:
                    if self._tool_rounds >= self._max_steps:
                        raise _MaxStepsExceeded("maximum tool steps exceeded")
                    self._tool_rounds += 1
                break
        return None

    async def awrap_tool_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> ToolMessage:
        async with self._lock:
            call = request.tool_call
            name = call.get("name")
            raw = call.get("args", {})
            effective_args = raw
            query_normalized = False
            # A model may keywordize or invent venue context for a retrieval
            # query.  Retrieval is only allowed to represent this request, so
            # replace that one field before both validation and MCP execution.
            # Other arguments (including top_k) remain model-controlled and
            # are still strictly validated below.
            if name == "search_knowledge" and isinstance(raw, dict):
                effective_args = {**raw, "query": self._user_message}
                query_normalized = raw.get("query") != self._user_message
            effective_request = (
                request.override(tool_call={**call, "args": effective_args})
                if effective_args is not raw
                else request
            )
            result: dict[str, object]
            if self.aborted:
                result = {
                    "status": "tool_error",
                    "tool_name": str(name),
                    "data": None,
                    "error": {"type": "aborted"},
                }
            elif not isinstance(name, str) or name not in {
                item.name for item in TOOL_DEFINITIONS
            }:
                result = {
                    "status": "unknown_tool",
                    "tool_name": str(name),
                    "data": None,
                    "error": {"type": "unknown_tool"},
                }
            else:
                definition = next(
                    item for item in TOOL_DEFINITIONS if item.name == name
                )
                try:
                    definition.arguments_model.model_validate(effective_args)
                    message = await asyncio.wait_for(
                        handler(effective_request), self._tool_timeout_seconds
                    )
                    artifact = getattr(message, "artifact", None)
                    parsed = (
                        artifact.get("structured_content")
                        if isinstance(artifact, dict)
                        else None
                    )
                    if parsed is None:
                        content = message.content
                        if (
                            isinstance(content, list)
                            and len(content) == 1
                            and isinstance(content[0], dict)
                        ):
                            content = content[0].get("text")
                        parsed = (
                            json.loads(content) if isinstance(content, str) else None
                        )
                    if (
                        not isinstance(parsed, dict)
                        or parsed.get("tool_name") != name
                        or parsed.get("status")
                        not in {"ok", "invalid_arguments", "unknown_tool", "tool_error"}
                    ):
                        raise ValueError("MCP result violates the tool envelope")
                    result = parsed
                except ValidationError:
                    result = {
                        "status": "invalid_arguments",
                        "tool_name": name,
                        "data": None,
                        "error": {"type": "validation_error"},
                    }
                except asyncio.CancelledError:
                    await self._client.ainvalidate()
                    raise
                except (
                    TimeoutError,
                    MCPProtocolError,
                    ToolException,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    self.aborted = True
                    self._abort_reason = "MCP protocol failure"
                    await self._client.ainvalidate()
                    result = {
                        "status": "tool_error",
                        "tool_name": name,
                        "data": None,
                        "error": {"type": "mcp_protocol_error"},
                    }
            status = result.get("status")
            trace_status = status if isinstance(status, str) else "invalid_result"
            payload = result.get("data")
            business_status = (
                payload.get("status")
                if isinstance(payload, dict) and isinstance(payload.get("status"), str)
                else None
            )
            self._traces.append(
                ToolTrace(
                    name if isinstance(name, str) else str(name),
                    effective_args if isinstance(effective_args, dict) else {},
                    result,
                    trace_status,
                    business_status,
                    query_normalized,
                )
            )
            return ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=str(call.get("id", "")),
                name=name if isinstance(name, str) else None,
                status="success" if trace_status == "ok" else "error",
            )

    async def awrap_model_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        if self.aborted:
            raise _ToolExecutionStopped(self._abort_reason)
        try:
            return await handler(request)
        except asyncio.CancelledError:
            raise
        except (
            APIConnectionError,
            APIError,
            APIStatusError,
            APITimeoutError,
            TimeoutError,
            ConnectionError,
        ) as error:
            raise _ModelFailure("model request failed") from error


class LangChainGuideAgent:
    """LangChain agent that discovers every business tool through MCP."""

    def __init__(
        self,
        model: BaseChatModel,
        mcp_client: MCPToolClient,
        *,
        max_steps: int = 4,
        scene_context: str | None = None,
        owns_model: bool = False,
        tool_timeout_seconds: float = 30,
    ) -> None:
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        if tool_timeout_seconds <= 0:
            raise ValueError("tool_timeout_seconds must be positive")
        self._model = model
        self._mcp_client = mcp_client
        self._max_steps = max_steps
        self._scene_context = scene_context
        self._owns_model = owns_model
        self._tool_timeout_seconds = tool_timeout_seconds
        self._closed = False

    @classmethod
    def from_settings(
        cls,
        settings: ModelSettings,
        mcp_client: MCPToolClient,
        *,
        max_steps: int = 4,
        scene_context: str | None = None,
    ) -> "LangChainGuideAgent":
        model = build_chat_model(settings)
        return cls(
            model,
            mcp_client,
            max_steps=max_steps,
            scene_context=scene_context,
            owns_model=True,
        )

    def _system_prompt(self, current_location: str | None = None) -> str:
        prompt = (
            DEVELOPER_INSTRUCTIONS
            if self._scene_context is None
            else f"{DEVELOPER_INSTRUCTIONS}\n\n{self._scene_context}"
        )
        if current_location:
            prompt += (
                f"\n\n本次会话由服务器确认的当前位置 POI ID 是 {current_location}。"
                "用户使用‘这里’‘当前位置’等表达时以此为起点；不要自行改变该位置。"
            )
        return prompt

    async def arun(
        self,
        user_message: str,
        *,
        history: tuple[dict[str, str], ...] = (),
        current_location: str | None = None,
    ) -> AgentResult:
        traces: list[ToolTrace] = []
        middleware = _MCPMiddleware(
            self._mcp_client,
            traces,
            self._max_steps,
            self._tool_timeout_seconds,
            user_message,
        )
        try:
            tools = await self._mcp_client.tools()
            graph = create_agent(
                self._model,
                tools,
                system_prompt=self._system_prompt(current_location),
                middleware=[middleware],
            )
            state = await graph.ainvoke(
                {
                    "messages": [
                        *history,
                        {"role": "user", "content": user_message},
                    ]
                },
                config={"recursion_limit": (self._max_steps * 4) + 12},
            )
        except asyncio.CancelledError:
            await self._mcp_client.ainvalidate()
            raise
        except MCPProtocolError as error:
            return AgentResult("tool_error", None, tuple(traces), str(error), None)
        except _MaxStepsExceeded:
            return AgentResult(
                "max_steps_exceeded",
                None,
                tuple(traces),
                "maximum tool steps exceeded",
                None,
            )
        except _ModelFailure as error:
            return AgentResult("model_error", None, tuple(traces), str(error), None)
        except _ToolExecutionStopped as error:
            return AgentResult("tool_error", None, tuple(traces), str(error), None)
        last_ai = next(
            (
                message
                for message in reversed(state.get("messages", []))
                if isinstance(message, AIMessage)
            ),
            None,
        )
        if last_ai is None:
            return AgentResult(
                "empty_response",
                None,
                tuple(traces),
                "model returned no assistant message",
                None,
            )
        text = (
            last_ai.content
            if isinstance(last_ai.content, str)
            else getattr(last_ai, "text", "")
        )
        metadata = getattr(last_ai, "response_metadata", {}) or {}
        response_id = metadata.get("response_id") or metadata.get("id")
        if not isinstance(text, str) or not text.strip():
            return AgentResult(
                "empty_response",
                None,
                tuple(traces),
                "model returned neither text nor tool calls",
                response_id if isinstance(response_id, str) else None,
            )
        return AgentResult(
            "completed",
            text,
            tuple(traces),
            None,
            response_id if isinstance(response_id, str) else None,
        )

    def run(
        self,
        user_message: str,
        *,
        history: tuple[dict[str, str], ...] = (),
        current_location: str | None = None,
    ) -> AgentResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.arun(
                    user_message,
                    history=history,
                    current_location=current_location,
                )
            )
        raise RuntimeError(
            "LangChainGuideAgent.run cannot be called from a running event loop; use arun"
        )

    async def aclose(self) -> None:
        await self._mcp_client.aclose()
        if not self._owns_model or self._closed:
            return
        self._closed = True
        for attribute in ("root_async_client", "root_client"):
            close = getattr(getattr(self._model, attribute, None), "close", None)
            if close is not None:
                value = close()
                if hasattr(value, "__await__"):
                    await value
