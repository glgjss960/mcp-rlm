from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import asyncio
import inspect
import json
import sys

from .mcp import MCPInvocationContext, MCPRegistry


@dataclass
class MCPServerInfo:
    name: str
    version: str


class StdioMCPServer:
    """Minimal MCP-compatible stdio server for tool calls used by MCP-RLM runtime."""

    def __init__(
        self,
        *,
        registry: MCPRegistry,
        server_info: Optional[MCPServerInfo] = None,
    ) -> None:
        self._registry = registry
        self._server_info = server_info or MCPServerInfo(name="mcp-rlm-server", version="0.1.0")

    async def serve_forever(self) -> None:
        while True:
            raw = await asyncio.to_thread(sys.stdin.buffer.readline)
            if not raw:
                return

            line = raw.decode("utf-8").strip()
            if not line:
                continue

            response = await self._handle_line(line)
            if response is None:
                continue

            out = json.dumps(response, ensure_ascii=False)
            sys.stdout.write(out + "\n")
            sys.stdout.flush()

    async def _handle_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {exc}",
                },
            }

        request_id = request.get("id")
        method = str(request.get("method", ""))
        params = request.get("params") or {}

        if not method:
            return self._error(request_id, -32600, "Invalid request: missing method")

        try:
            result = await self._dispatch(method=method, params=params)
            if request_id is None:
                return None
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }
        except Exception as exc:
            return self._error(request_id, -32000, str(exc))

    async def _dispatch(self, *, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "initialize":
            return {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": self._server_info.name,
                    "version": self._server_info.version,
                },
            }

        if method == "ping":
            return {"ok": True}

        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": name,
                        "description": f"Tool '{name}' exposed by MCP-RLM server",
                        "inputSchema": {"type": "object"},
                    }
                    for name in self._registry.list_objects()
                ]
            }

        if method == "tools/call":
            tool_name = str(params.get("name", ""))
            arguments = params.get("arguments") or {}
            context = params.get("context") or {}

            if not tool_name:
                raise RuntimeError("tools/call requires non-empty tool name")

            handler = self._registry.get(tool_name)
            invocation_ctx = MCPInvocationContext(
                episode_id=str(context.get("episode_id", "")),
                group_id=str(context.get("group_id", "")),
            )

            raw = handler(arguments, invocation_ctx)
            if inspect.isawaitable(raw):
                output = await raw
            else:
                output = raw

            return {
                "isError": False,
                "result": output,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(output, ensure_ascii=False),
                    }
                ],
            }

        raise RuntimeError(f"Unsupported MCP method: {method}")

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
