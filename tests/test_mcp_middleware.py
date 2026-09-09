import asyncio
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage

from guide_agent.langchain_agent import _MCPMiddleware


@dataclass(frozen=True)
class FakeToolCallRequest:
    tool_call: dict[str, Any]

    def override(self, **changes: Any) -> "FakeToolCallRequest":
        return FakeToolCallRequest(
            tool_call=changes.get("tool_call", self.tool_call)
        )


class UnusedClient:
    async def ainvalidate(self) -> None:
        raise AssertionError("the client should not be invalidated")


def _ok_message(call: dict[str, Any]) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(
            {
                "status": "ok",
                "tool_name": call["name"],
                "data": [],
                "error": None,
            }
        ),
        tool_call_id=call["id"],
        name=call["name"],
    )


def test_search_query_is_forced_to_the_cleaned_original_user_message() -> None:
    async def exercise() -> None:
        traces = []
        middleware = _MCPMiddleware(
            UnusedClient(), traces, max_steps=4, timeout=1, user_message="  这里可以停车吗？  "
        )
        request = FakeToolCallRequest(
            {
                "id": "call-1",
                "name": "search_knowledge",
                "args": {"query": "星河科技体验中心停车信息", "top_k": 3},
            }
        )

        async def handler(effective_request: FakeToolCallRequest) -> ToolMessage:
            assert effective_request.tool_call["args"] == {
                "query": "这里可以停车吗？",
                "top_k": 3,
            }
            return _ok_message(effective_request.tool_call)

        await middleware.awrap_tool_call(request, handler)

        assert traces[0].arguments == {"query": "这里可以停车吗？", "top_k": 3}
        assert traces[0].query_normalized is True

    asyncio.run(exercise())


def test_non_search_tool_arguments_are_not_modified() -> None:
    async def exercise() -> None:
        traces = []
        middleware = _MCPMiddleware(
            UnusedClient(), traces, max_steps=4, timeout=1, user_message="这里可以停车吗？"
        )
        request = FakeToolCallRequest(
            {
                "id": "call-2",
                "name": "plan_route",
                "args": {"start_id": "entrance", "end_id": "ai_lab"},
            }
        )

        async def handler(effective_request: FakeToolCallRequest) -> ToolMessage:
            assert effective_request is request
            assert effective_request.tool_call["args"] == request.tool_call["args"]
            return _ok_message(effective_request.tool_call)

        await middleware.awrap_tool_call(request, handler)

        assert traces[0].arguments == {"start_id": "entrance", "end_id": "ai_lab"}
        assert traces[0].query_normalized is False

    asyncio.run(exercise())
