"""Composition root for the public LangChain + MCP runtime."""

import os
from pathlib import Path

from guide_agent.answering import ChatService, DEFAULT_DEMO_MINIMUM_SCORE
from guide_agent.documents import DocumentLoadError
from guide_agent.langchain_agent import LangChainGuideAgent
from guide_agent.mcp_client import MCPToolClient
from guide_agent.model_settings import ModelConfigurationError, load_model_settings
from guide_agent.scene import Scene, SceneLoadError, load_scene

PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_SCENE_PATH = PROJECT_ROOT / "scenes" / "demo" / "scene.json"


class RuntimeBuildError(RuntimeError):
    """The runtime could not be safely initialized."""


def build_default_chat_service(scene_path: str | Path | None = None) -> ChatService:
    """Build FastAPI/CLI's single supported architecture.

    LangChain owns the model loop; the three domain tools are loaded from the
    task-owned stdio MCP server, never directly from the web process.
    """
    resolved = resolve_scene_path(scene_path)
    try:
        settings = load_model_settings()
        scene = load_scene(resolved)
        agent = LangChainGuideAgent.from_settings(
            settings,
            MCPToolClient(resolved),
            scene_context=_build_scene_context(scene),
        )
        return ChatService(agent, minimum_score=DEFAULT_DEMO_MINIMUM_SCORE)
    except (ModelConfigurationError, SceneLoadError, DocumentLoadError, OSError) as error:
        raise RuntimeBuildError("failed to initialize LangChain + MCP runtime") from error


def resolve_scene_path(scene_path: str | Path | None = None) -> Path:
    value = scene_path if scene_path is not None else os.environ.get("GUIDE_SCENE_PATH", "").strip()
    candidate = Path(value) if value else DEFAULT_SCENE_PATH
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _build_scene_context(scene: Scene) -> str:
    poi_ids = ", ".join(poi.id for poi in scene.pois)
    poi_labels = "; ".join(
        f"{poi.id} = {poi.name}"
        + (f"（别名：{', '.join(poi.aliases)}）" if poi.aliases else "")
        for poi in scene.pois
    )
    return (
        f"场景系统提示：{scene.system_prompt}\n"
        f"POI 工具参数允许的纯 ID 清单：{poi_ids}\n"
        f"名称和别名映射（只用于理解用户问题，不能原样作为工具参数）：{poi_labels}\n"
        "工具参数硬性规则：lookup_poi 的 poi_id，以及 plan_route 的 start_id、end_id，"
        "必须且只能填写上面清单中的一个精确纯 ID（例如 entrance、ai_lab）。"
        "绝不能填写名称、别名、冒号后的说明、方括号注释，或把它们拼接进 ID。\n"
        "知识检索规则：调用 search_knowledge 时，query 必须保留用户完整原问题和场馆语境，"
        "不要缩写为孤立关键词（例如不能把“体验中心常规开放时间是什么？”缩成“常规开放时间”）。\n"
        "禁止猜测 POI ID、路线距离、开放时间和设施属性；不确定时调用工具或请求澄清。"
    )
