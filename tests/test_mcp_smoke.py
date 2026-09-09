import pytest

from guide_agent.mcp_client import MCPToolClient
from guide_agent.runtime import DEFAULT_SCENE_PATH


@pytest.mark.anyio
async def test_mcp_server_exposes_only_the_three_business_tools() -> None:
    client = MCPToolClient(DEFAULT_SCENE_PATH, startup_timeout_seconds=45)
    try:
        tools = await client.tools()
        assert {tool.name for tool in tools} == {"search_knowledge", "lookup_poi", "plan_route"}
    finally:
        await client.aclose()
