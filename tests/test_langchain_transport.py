"""Offline checks for the real LangChain provider serialization."""

import asyncio
import json

import httpx
import pytest

from guide_agent.langchain_agent import build_chat_model
from guide_agent.model_settings import ModelSettings


def _responses_reply(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1/responses"
    body = json.loads(request.content)
    assert body["model"] == "offline-model"
    assert body["input"]
    return httpx.Response(
        200,
        json={
            "id": "resp_offline_1",
            "object": "response",
            "created_at": 1,
            "model": "offline-model",
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "output": [
                {
                    "id": "message_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "responses ok",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": False,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    )


def _chat_completions_reply(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1/chat/completions"
    body = json.loads(request.content)
    assert body["model"] == "offline-model"
    assert body["messages"]
    return httpx.Response(
        200,
        json={
            "id": "chat_offline_1",
            "object": "chat.completion",
            "created": 1,
            "model": "offline-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "chat completions ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.mark.parametrize(
    ("api_format", "reply", "expected_text"),
    [
        ("responses", _responses_reply, "responses ok"),
        ("chat_completions", _chat_completions_reply, "chat completions ok"),
    ],
)
def test_langchain_model_factory_targets_the_configured_endpoint(
    api_format: str,
    reply,
    expected_text: str,
) -> None:
    transport = httpx.MockTransport(reply)

    async def invoke() -> str:
        with httpx.Client(transport=transport) as sync_client:
            async with httpx.AsyncClient(transport=transport) as async_client:
                model = build_chat_model(
                    ModelSettings(
                        model="offline-model",
                        api_key="offline-not-a-secret",
                        base_url="http://offline.test/v1",
                        api_format=api_format,
                    ),
                    http_client=sync_client,
                    http_async_client=async_client,
                )
                response = await model.ainvoke("hello")
                return response.text

    assert asyncio.run(invoke()) == expected_text
