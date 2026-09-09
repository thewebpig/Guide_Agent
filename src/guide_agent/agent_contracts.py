"""Stable result contracts for the LangChain + MCP agent."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolTrace:
    """One auditable tool execution."""

    tool_name: str
    arguments: dict[str, object]
    result: dict[str, object]
    status: str


@dataclass(frozen=True)
class AgentResult:
    """The stable result returned by every agent backend."""

    status: str
    answer: str | None
    tool_traces: tuple[ToolTrace, ...]
    error: str | None
    response_id: str | None


class AgentRunner(Protocol):
    def run(self, user_message: str) -> AgentResult: ...

    async def arun(self, user_message: str) -> AgentResult: ...

    async def aclose(self) -> None: ...
