"""The isolated stdio MCP server for the three audited guide tools.

Nothing in this module writes to stdout except the MCP transport.  In
particular, model configuration is deliberately not loaded in this process.
"""

from __future__ import annotations

import asyncio
import logging
import os
import hashlib
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from pydantic import ConfigDict

from guide_agent.retrieval import build_scene_knowledge_index
from guide_agent.scene import load_scene
from guide_agent.service import GuideService
from guide_agent.tools import ToolRegistry

LOG = logging.getLogger(__name__)


def scene_fingerprint(scene_path: str | Path) -> str:
    path = Path(scene_path).resolve()
    scene = load_scene(path)
    digest = hashlib.sha256()
    for item in (path, *(path.parent / source for source in scene.source_docs)):
        digest.update(str(item.resolve()).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def build_server(scene_path: str | Path) -> FastMCP:
    """Build a server and its one-per-process index.

    ``embedder`` is intentionally not configurable through the environment:
    test code may inject one by calling the lower-level registry factory in its
    own process, while production has no fake-embedding switch.
    """
    path = Path(scene_path).resolve()
    scene = load_scene(path)
    registry = ToolRegistry(GuideService(scene), build_scene_knowledge_index(path))
    return build_server_for_registry(registry, fingerprint=scene_fingerprint(path))


class _WireArguments(ArgModelBase):
    """Keep every argument intact until ToolRegistry validates it."""
    model_config = ConfigDict(extra="allow")

    def model_dump_one_level(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class _RegistryMetadata(FuncMetadata):
    def pre_parse_json(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


def build_server_for_registry(registry: ToolRegistry, *, fingerprint: str = "test-fixture") -> FastMCP:
    """Expose an already-built registry; useful for isolated test processes."""
    server = FastMCP("guide-agent-tools")

    @server.resource("guide://scene-fingerprint")
    def scene_hash() -> str:
        return fingerprint

    def register(name: str) -> None:
        definition = next(item for item in registry.schemas() if item["name"] == name)

        async def tool(**arguments: Any) -> dict[str, object]:
            return await asyncio.to_thread(registry.execute, name, arguments)

        server.tool(name=name, description=str(definition["description"]), structured_output=True)(tool)
        # The SDK normally coerces stringified JSON and ignores extra fields.
        # Preserve raw wire arguments for the registry's single validation path.
        # This narrow SDK registration seam is covered by real stdio tests.
        registered = server._tool_manager.get_tool(name)
        registered.parameters = definition["parameters"]
        metadata = registered.fn_metadata
        registered.fn_metadata = _RegistryMetadata(**{
            key: getattr(metadata, key) for key in type(metadata).model_fields
        })
        registered.fn_metadata.arg_model = _WireArguments

    for tool_name in ("search_knowledge", "lookup_poi", "plan_route"):
        register(tool_name)
    return server


def main() -> None:
    scene_path = os.environ.get("GUIDE_SCENE_PATH", "").strip()
    if not scene_path:
        raise RuntimeError("GUIDE_SCENE_PATH is required for the MCP server")
    logging.basicConfig(level=logging.INFO)
    # FastMCP owns stdio and keeps protocol bytes off application stdout.
    with redirect_stdout(sys.stderr):
        server = build_server(scene_path)
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
