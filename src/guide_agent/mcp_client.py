"""Persistent, task-owned stdio MCP client."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from guide_agent.tools import TOOL_DEFINITIONS
from guide_agent.mcp_server import scene_fingerprint


class MCPProtocolError(RuntimeError):
    """The server/session violated the client protocol contract."""


_ALLOWED = {definition.name: definition.arguments_model.model_json_schema() for definition in TOOL_DEFINITIONS}
_SAFE_ENV = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    # A request may be cancelled while the owner is still unwinding. Retrieving
    # its eventual exception prevents a detached-future warning, while awaiters
    # still receive that same exception normally.
    if not future.cancelled():
        future.exception()


class MCPToolClient:
    """One subprocess session, entered and exited by one owner task.

    Async context managers from the MCP SDK carry AnyIO cancel scopes. Keeping
    them wholly inside ``_owner`` avoids exiting a scope from FastAPI's request
    task during cancellation or application shutdown.
    """
    def __init__(self, scene_path: str | Path, *, timeout_seconds: float = 30, server_parameters: StdioServerParameters | None = None, startup_timeout_seconds: float | None = None) -> None:
        if timeout_seconds <= 0 or (startup_timeout_seconds is not None and startup_timeout_seconds <= 0):
            raise ValueError("MCP deadlines must be positive")
        self._scene_path = str(Path(scene_path).resolve())
        self._timeout = timeout_seconds
        # A process import and MCP handshake can legitimately take longer than
        # an individual tool request.  Keep the latter configurable without
        # making a short request deadline spuriously fail startup.
        self._startup_timeout = startup_timeout_seconds if startup_timeout_seconds is not None else max(timeout_seconds, 10.0)
        self._fingerprint = scene_fingerprint(self._scene_path)
        self._server_parameters = server_parameters
        self._owner_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[list[BaseTool]] | None = None
        self._stop: asyncio.Event | None = None
        self._requests: asyncio.Queue[tuple[str, tuple[Any, ...], dict[str, Any], asyncio.Future[Any]]] | None = None
        self._bootstrap_session: ClientSession | None = None
        self._closing = False
        self._lock = asyncio.Lock()
        self._call_lock = asyncio.Lock()

    def _environment(self) -> dict[str, str]:
        env = {key: os.environ[key] for key in _SAFE_ENV if key in os.environ}
        env["GUIDE_SCENE_PATH"] = self._scene_path
        return env

    async def tools(self) -> list[BaseTool]:
        async with self._lock:
            if self._closing:
                raise MCPProtocolError("MCP client is closed")
            if self._owner_task is None or self._owner_task.done():
                self._ready = asyncio.get_running_loop().create_future()
                self._ready.add_done_callback(_consume_future_exception)
                self._stop = asyncio.Event()
                self._requests = asyncio.Queue()
                self._owner_task = asyncio.create_task(self._owner(self._ready, self._stop))
            ready = self._ready
            owner = self._owner_task
        try:
            return await asyncio.wait_for(asyncio.shield(ready), self._startup_timeout)
        except asyncio.CancelledError:
            await self._invalidate(expected_task=owner)
            raise
        except BaseException:
            await self._invalidate(expected_task=owner)
            raise MCPProtocolError("MCP transport failure") from None

    async def _owner(self, ready: asyncio.Future[list[BaseTool]], stop: asyncio.Event) -> None:
        try:
            params = self._server_parameters or StdioServerParameters(command=sys.executable, args=["-m", "guide_agent.mcp_server"], env=self._environment(), cwd=str(Path(self._scene_path).parent))
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), self._startup_timeout)
                    resource = await asyncio.wait_for(session.read_resource("guide://scene-fingerprint"), self._startup_timeout)
                    contents = getattr(resource, "contents", [])
                    actual = contents[0].text if len(contents) == 1 and isinstance(getattr(contents[0], "text", None), str) else None
                    if actual != self._fingerprint:
                        raise MCPProtocolError("MCP scene fingerprint mismatch")
                    self._bootstrap_session = session
                    # The adapter must be used to discover and construct the
                    # LangChain tools.  The proxy schedules every later
                    # call_tool back onto this owner task.
                    tools = await asyncio.wait_for(load_mcp_tools(_SessionProxy(self), handle_tool_errors=False), self._startup_timeout)
                    self._validate_tools(tools)
                    self._bootstrap_session = None
                    ready.set_result(tools)
                    while not stop.is_set():
                        request = await self._next_request(stop)
                        if request is None:
                            continue
                        method, args, kwargs, response = request
                        if response.cancelled():
                            continue
                        try:
                            value = await asyncio.wait_for(getattr(session, method)(*args, **kwargs), self._timeout)
                        except asyncio.CancelledError:
                            response.cancel()
                            raise
                        except Exception:
                            if not response.done():
                                response.set_exception(MCPProtocolError("MCP call failed"))
                            break
                        else:
                            if not response.done():
                                response.set_result(value)
        except BaseException as error:
            if not ready.done():
                ready.set_exception(MCPProtocolError("MCP startup or protocol failure"))
        finally:
            self._fail_pending()

    def _fail_pending(self) -> None:
        if self._requests is None:
            return
        while not self._requests.empty():
            _, _, _, response = self._requests.get_nowait()
            if not response.done():
                response.set_exception(MCPProtocolError("MCP session closed"))

    async def _next_request(self, stop: asyncio.Event) -> tuple[str, tuple[Any, ...], dict[str, Any], asyncio.Future[Any]] | None:
        assert self._requests is not None
        get_request = asyncio.create_task(self._requests.get())
        get_stop = asyncio.create_task(stop.wait())
        try:
            done, pending = await asyncio.wait((get_request, get_stop), return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            return get_request.result() if get_request in done else None
        finally:
            for task in (get_request, get_stop):
                if not task.done():
                    task.cancel()
            await asyncio.gather(get_request, get_stop, return_exceptions=True)

    async def _submit(self, method: str, *args: Any, **kwargs: Any) -> Any:
        # Serialize protocol calls as well as model tool ordering. Waiting
        # callers do not start a new session while a failed call is torn down.
        async with self._call_lock:
            await self.tools()
            queue, owner = self._requests, self._owner_task
            if queue is None:
                raise MCPProtocolError("MCP session is unavailable")
            response: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            response.add_done_callback(_consume_future_exception)
            await queue.put((method, args, kwargs, response))
            try:
                return await asyncio.wait_for(asyncio.shield(response), self._timeout)
            except asyncio.CancelledError:
                await self._invalidate(expected_task=owner)
                raise
            except Exception:
                await self._invalidate(expected_task=owner)
                raise MCPProtocolError("MCP call failed") from None
            finally:
                if not response.done():
                    response.cancel()

    @staticmethod
    def _validate_tools(tools: list[BaseTool]) -> None:
        actual = {tool.name: tool for tool in tools}
        if len(tools) != len(_ALLOWED) or set(actual) != set(_ALLOWED):
            raise MCPProtocolError("MCP tool allowlist mismatch")
        for name, expected in _ALLOWED.items():
            schema = getattr(actual[name], "args_schema", None)
            exported = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
            # title differences are SDK metadata, not parameter semantics.
            if not isinstance(exported, dict) or exported.get("properties") != expected.get("properties") or set(exported.get("required", [])) != set(expected.get("required", [])) or exported.get("additionalProperties") is not False:
                raise MCPProtocolError(f"MCP schema mismatch for {name}")

    async def _invalidate(self, *, expected_task: asyncio.Task[None] | None = None) -> None:
        # Several request waiters may time out or be cancelled together.  One
        # teardown owns the task/context managers; a second teardown must not
        # clear state belonging to a newly created owner.
        async with self._lock:
            if expected_task is not None and self._owner_task is not expected_task:
                return
            task = self._owner_task
            stop = self._stop
            if stop is not None:
                stop.set()
            if task is not None and task is not asyncio.current_task():
                try:
                    await asyncio.wait_for(asyncio.shield(task), self._timeout)
                except TimeoutError:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                except (Exception, asyncio.CancelledError):
                    pass
            self._fail_pending()
            # Do not detach state installed by a new owner while this older
            # owner was being awaited.
            if self._owner_task is task:
                self._owner_task = None
                self._ready = None
                self._stop = None
                self._requests = None
                self._bootstrap_session = None

    async def aclose(self) -> None:
        self._closing = True
        await self._invalidate()

    async def ainvalidate(self) -> None:
        """Drop a broken connection; a later request may create a new one."""
        await self._invalidate()


class _SessionProxy:
    """Only the MCP SDK calls needed by ``load_mcp_tools``."""
    def __init__(self, client: MCPToolClient) -> None:
        self._client = client

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        if self._client._bootstrap_session is not None:
            return await self._client._bootstrap_session.list_tools(*args, **kwargs)
        return await self._client._submit("list_tools", *args, **kwargs)

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        return await self._client._submit("call_tool", *args, **kwargs)
