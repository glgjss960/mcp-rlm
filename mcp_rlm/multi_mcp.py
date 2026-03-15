from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import asyncio

from .mcp import MCPCall, MCPInvocationContext, MCPResult
from .stdio_mcp_client import StdioMCPClient


@dataclass
class MCPServerSpec:
    alias: str
    command: Sequence[str]
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    max_concurrency: int = 32


class MultiServerMCPClient:
    """Route MCP tool calls across multiple stdio MCP servers.

    Object names use the format: '<alias>/<tool_name>'.
    """

    def __init__(self, specs: List[MCPServerSpec], *, default_alias: Optional[str] = None) -> None:
        if not specs:
            raise ValueError("specs must not be empty")

        self._clients: Dict[str, StdioMCPClient] = {}
        for spec in specs:
            if spec.alias in self._clients:
                raise ValueError(f"Duplicate alias: {spec.alias}")
            self._clients[spec.alias] = StdioMCPClient(
                command=list(spec.command),
                cwd=spec.cwd,
                env=spec.env,
                max_concurrency=spec.max_concurrency,
            )

        self.default_alias = default_alias
        if self.default_alias and self.default_alias not in self._clients:
            raise ValueError(f"default_alias not found in specs: {self.default_alias}")

    async def start(self) -> None:
        await asyncio.gather(*(client.start() for client in self._clients.values()))

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self._clients.values()), return_exceptions=True)

    async def __aenter__(self) -> "MultiServerMCPClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def list_objects(self) -> List[str]:
        out: List[str] = []
        for alias, client in self._clients.items():
            tools = await client.list_tools()
            out.extend([f"{alias}/{tool}" for tool in tools])
        return sorted(out)

    async def call(self, call: MCPCall, ctx: MCPInvocationContext) -> MCPResult:
        alias, tool_name = self._route(call.object_name)
        routed = MCPCall(object_name=tool_name, payload=call.payload, timeout_seconds=call.timeout_seconds)
        return await self._clients[alias].call(routed, ctx)

    async def call_many(self, calls: List[MCPCall], ctx: MCPInvocationContext) -> List[MCPResult]:
        tasks = [asyncio.create_task(self.call(call, ctx)) for call in calls]
        return await asyncio.gather(*tasks)

    def _route(self, object_name: str) -> tuple[str, str]:
        if "/" in object_name:
            alias, tool_name = object_name.split("/", 1)
            if alias not in self._clients:
                raise KeyError(f"Unknown MCP server alias: {alias}")
            if not tool_name:
                raise ValueError(f"Invalid object name: {object_name}")
            return alias, tool_name

        if self.default_alias is None:
            raise ValueError(
                f"Object name '{object_name}' has no alias and no default_alias configured. Use '<alias>/<tool>'"
            )
        return self.default_alias, object_name
