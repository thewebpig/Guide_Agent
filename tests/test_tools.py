from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from guide_agent.documents import DocumentChunk
from guide_agent.retrieval import build_knowledge_index
from guide_agent.scene import load_scene
from guide_agent.service import GuideService
from guide_agent.tools import (
    LookupPoiArguments,
    PlanRouteArguments,
    SearchKnowledgeArguments,
    ToolRegistry,
    get_tool_schemas,
)
from guide_agent.retrieval import (
    KnowledgeSearchError,
    build_knowledge_index,
)


PROJECT_ROOT = Path(__file__).parents[1]
DEMO_SCENE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "demo_scene"
    / "scene.json"
)


class FakeEmbedder:
    """为Tool测试提供固定向量。"""

    def embed(
        self,
        documents: list[str],
    ) -> Iterable[NDArray[np.float32]]:
        return iter([
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
            for _ in documents
        ])

    def query_embed(
        self,
        queries: list[str],
    ) -> Iterable[NDArray[np.float32]]:
        return iter([
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
            for _ in queries
        ])


@pytest.fixture
def tool_registry() -> ToolRegistry:
    scene = load_scene(DEMO_SCENE_PATH)
    service = GuideService(scene)

    knowledge_index = build_knowledge_index(
        [
            DocumentChunk(
                chunk_id="demo.md::chunk-0001",
                text="场馆开放时间为9:30。",
                source="demo.md",
            )
        ],
        FakeEmbedder(),
    )

    return ToolRegistry(
        service,
        knowledge_index,
    )


def test_lookup_poi_arguments_accept_valid_input() -> None:
    arguments = LookupPoiArguments.model_validate(
        {"poi_id": "entrance"}
    )

    assert arguments.poi_id == "entrance"


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"poi_id": 123},
        {"poi_id": ""},
        {"poi_id": "   "},
        {
            "poi_id": "entrance",
            "unexpected": True,
        },
    ],
)
def test_lookup_poi_arguments_reject_invalid_input(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        LookupPoiArguments.model_validate(arguments)


def test_plan_route_arguments_clean_valid_input() -> None:
    arguments = PlanRouteArguments.model_validate(
        {
            "start_id": " entrance ",
            "end_id": " exhibition ",
        }
    )

    assert arguments.model_dump() == {
        "start_id": "entrance",
        "end_id": "exhibition",
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"end_id": "exhibition"},
        {"start_id": "entrance"},
        {
            "start_id": "   ",
            "end_id": "exhibition",
        },
        {
            "start_id": "entrance",
            "end_id": 123,
        },
    ],
)
def test_plan_route_arguments_reject_invalid_input(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        PlanRouteArguments.model_validate(arguments)


def test_search_knowledge_arguments_accept_explicit_top_k() -> None:
    arguments = SearchKnowledgeArguments.model_validate(
        {"query": " 开放时间 ", "top_k": 5}
    )

    assert arguments.model_dump() == {
        "query": "开放时间",
        "top_k": 5,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"query": "开放时间"},
        {"query": "   "},
        {"query": "开放时间", "top_k": 0},
        {"query": "开放时间", "top_k": -1},
        {"query": "开放时间", "top_k": True},
        {"query": "开放时间", "top_k": 1.5},
        {"query": "开放时间", "top_k": "5"},
    ],
)
def test_search_knowledge_arguments_reject_invalid_input(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SearchKnowledgeArguments.model_validate(arguments)


def test_get_tool_schemas_returns_three_registered_tools() -> None:
    schemas = get_tool_schemas()

    assert [
        schema["name"]
        for schema in schemas
    ] == [
        "search_knowledge",
        "lookup_poi",
        "plan_route",
    ]


def test_lookup_poi_schema_describes_required_arguments() -> None:
    schemas = get_tool_schemas()
    lookup_schema = schemas[1]
    parameters = lookup_schema["parameters"]

    assert isinstance(parameters, dict)
    assert parameters["type"] == "object"
    assert parameters["required"] == ["poi_id"]
    assert parameters["additionalProperties"] is False
    assert (
        parameters["properties"]["poi_id"]["type"]
        == "string"
    )


def test_tool_registry_rejects_unknown_tool(
    tool_registry: ToolRegistry,
) -> None:
    result = tool_registry.execute(
        "run_shell",
        {},
    )

    assert result["status"] == "unknown_tool"
    assert result["data"] is None


def test_tool_registry_rejects_invalid_arguments(
    tool_registry: ToolRegistry,
) -> None:
    result = tool_registry.execute(
        "lookup_poi",
        {},
    )

    assert result["status"] == "invalid_arguments"
    assert result["data"] is None


def test_lookup_poi_tool_returns_business_result(
    tool_registry: ToolRegistry,
) -> None:
    result = tool_registry.execute(
        "lookup_poi",
        {"poi_id": "entrance"},
    )

    assert result["status"] == "ok"
    assert result["data"]["status"] == "ok"
    assert result["data"]["poi"]["id"] == "entrance"


def test_lookup_poi_tool_preserves_not_found(
    tool_registry: ToolRegistry,
) -> None:
    result = tool_registry.execute(
        "lookup_poi",
        {"poi_id": "unknown"},
    )

    assert result["status"] == "ok"
    assert result["data"]["status"] == "not_found"


def test_plan_route_tool_returns_route(
    tool_registry: ToolRegistry,
) -> None:
    result = tool_registry.execute(
        "plan_route",
        {
            "start_id": "entrance",
            "end_id": "exhibition",
        },
    )

    assert result["status"] == "ok"
    assert result["data"] == {
        "status": "ok",
        "path": [
            "entrance",
            "robot",
            "exhibition",
        ],
        "distance": 9.0,
        "reason": None,
    }


def test_plan_route_tool_preserves_unreachable(
    tool_registry: ToolRegistry,
) -> None:
    result = tool_registry.execute(
        "plan_route",
        {
            "start_id": "entrance",
            "end_id": "maintenance",
        },
    )

    assert result["status"] == "ok"
    assert result["data"]["status"] == "unreachable"


def test_plan_route_tool_preserves_invalid_poi(
    tool_registry: ToolRegistry,
) -> None:
    result = tool_registry.execute(
        "plan_route",
        {
            "start_id": "entrance",
            "end_id": "unknown",
        },
    )

    assert result["status"] == "ok"
    assert result["data"]["status"] == "invalid_input"


def test_search_knowledge_tool_returns_evidence(
    tool_registry: ToolRegistry,
) -> None:
    result = tool_registry.execute(
        "search_knowledge",
        {
            "query": "开放时间",
            "top_k": 1,
        },
    )

    assert result["status"] == "ok"
    assert result["data"] == [
        {
            "chunk_id": "demo.md::chunk-0001",
            "text": "场馆开放时间为9:30。",
            "source": "demo.md",
            "score": pytest.approx(1.0),
        }
    ]


def test_search_tool_converts_expected_error(
    tool_registry: ToolRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_search_error(
        *args: object,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        raise KnowledgeSearchError(
            "knowledge index is unavailable"
        )

    monkeypatch.setattr(
        "guide_agent.tools.search_knowledge",
        raise_search_error,
    )

    result = tool_registry.execute(
        "search_knowledge",
        {"query": "开放时间", "top_k": 1},
    )

    assert result == {
        "status": "tool_error",
        "tool_name": "search_knowledge",
        "data": None,
        "error": {
            "type": "knowledge_search_error",
            "message": "knowledge index is unavailable",
        },
    }
