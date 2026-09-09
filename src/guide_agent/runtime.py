"""Composition root for the public LangChain + MCP runtime."""

import os
from pathlib import Path

from guide_agent.answering import ChatService
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
        return ChatService(agent, minimum_score=0.5)
    except (ModelConfigurationError, SceneLoadError, DocumentLoadError, OSError) as error:
        raise RuntimeBuildError("failed to initialize LangChain + MCP runtime") from error


def resolve_scene_path(scene_path: str | Path | None = None) -> Path:
    value = scene_path if scene_path is not None else os.environ.get("GUIDE_SCENE_PATH", "").strip()
    candidate = Path(value) if value else DEFAULT_SCENE_PATH
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _build_scene_context(scene: Scene) -> str:
    pois = ", ".join(f"{poi.id}:{poi.name}[{','.join(poi.aliases)}]" if poi.aliases else f"{poi.id}:{poi.name}" for poi in scene.pois)
    return (
        f"场景系统提示：{scene.system_prompt}\n已知 POI ID/名称/别名：{pois}\n"
        "禁止猜测 POI ID、路线距离、开放时间和设施属性；不确定时调用工具或请求澄清。"
    )
