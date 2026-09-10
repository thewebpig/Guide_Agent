import asyncio
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage

from guide_agent.langchain_agent import (
    _DuplicateToolCall,
    _MCPMiddleware,
    _requires_tool_grounding,
)


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


def test_only_social_turns_can_skip_grounding_tools() -> None:
    assert _requires_tool_grounding("你好") is False
    assert _requires_tool_grounding("谢谢！") is False
    assert _requires_tool_grounding("杨善林院士的办公室在哪里？") is True
    assert _requires_tool_grounding("从这里去306怎么走？") is True


def test_duplicate_effective_tool_call_is_stopped_before_execution() -> None:
    async def exercise() -> None:
        traces = []
        middleware = _MCPMiddleware(
            UnusedClient(), traces, max_steps=4, timeout=1, user_message="访客可以把车停在哪里？"
        )
        request = FakeToolCallRequest(
            {
                "id": "call-1",
                "name": "search_knowledge",
                "args": {"query": "访客可以把车停在哪里？", "top_k": 5},
            }
        )
        calls = 0

        async def handler(effective_request: FakeToolCallRequest) -> ToolMessage:
            nonlocal calls
            calls += 1
            return _ok_message(effective_request.tool_call)

        await middleware.awrap_tool_call(request, handler)
        duplicate = FakeToolCallRequest(
            {**request.tool_call, "id": "call-2"}
        )
        with pytest.raises(_DuplicateToolCall):
            await middleware.awrap_tool_call(duplicate, handler)

        assert calls == 1
        assert len(traces) == 1

    import pytest

    asyncio.run(exercise())
