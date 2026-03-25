from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import asyncio
import inspect
import json
import sys

from .mcp import MCPInvocationContext, MCPRegistry
from .mcp_sdk import ensure_mcp_sdk, mcp_sdk_status

_SDK_AVAILABLE = False
_SDK_IMPORT_ERROR: Optional[Exception] = None

try:
    ensure_mcp_sdk()
    from mcp import types
    from mcp.server import Server, ServerRequestContext
    from mcp.server.stdio import stdio_server

    _SDK_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    types = Any  # type: ignore[assignment,misc]
    Server = Any  # type: ignore[assignment,misc]
    ServerRequestContext = Any  # type: ignore[assignment,misc]
    stdio_server = None  # type: ignore[assignment]
    _SDK_IMPORT_ERROR = exc


@dataclass
class MCPServerInfo:
    name: str
    version: str


class StdioMCPServer:
    """Stdio MCP server.

    Uses official MCP Python SDK when available; otherwise falls back to legacy JSON-RPC transport.
    """

    def __init__(
        self,
        *,
        registry: MCPRegistry,
        server_info: Optional[MCPServerInfo] = None,
        prefer_official_sdk: bool = True,
        strict_official_sdk: bool = False,
    ) -> None:
        if strict_official_sdk and not prefer_official_sdk:
            raise ValueError('strict_official_sdk=True requires prefer_official_sdk=True')

        self._registry = registry
        self._server_info = server_info or MCPServerInfo(name='mcp-rlm-server', version='0.1.0')
        self._use_sdk = bool(prefer_official_sdk and _SDK_AVAILABLE)

        if strict_official_sdk and not self._use_sdk:
            ok, detail = mcp_sdk_status()
            reason = detail if not ok else 'MCP SDK is available but not selected'
            if _SDK_IMPORT_ERROR is not None:
                reason = f'{reason}; import_error={type(_SDK_IMPORT_ERROR).__name__}: {_SDK_IMPORT_ERROR}'
            raise RuntimeError(f'Official MCP SDK is required but unavailable. {reason}')

        self._server: Optional[Server] = None
        if self._use_sdk:
            self._server = Server(
                self._server_info.name,
                version=self._server_info.version,
                on_list_tools=self._handle_list_tools,
                on_call_tool=self._handle_call_tool,
            )

    @property
    def using_official_sdk(self) -> bool:
        return self._use_sdk

    async def serve_forever(self) -> None:
        if self._use_sdk:
            await self._serve_forever_sdk()
        else:
            await self._serve_forever_legacy()

    async def _serve_forever_sdk(self) -> None:
        assert self._server is not None
        async with stdio_server() as streams:
            await self._server.run(
                streams[0],
                streams[1],
                self._server.create_initialization_options(),
            )

    async def _serve_forever_legacy(self) -> None:
        while True:
            raw = await asyncio.to_thread(sys.stdin.buffer.readline)
            if not raw:
                return

            line = raw.decode('utf-8').strip()
            if not line:
                continue

            response = await self._handle_line(line)
            if response is None:
                continue

            out = json.dumps(response, ensure_ascii=False)
            sys.stdout.write(out + '\n')
            sys.stdout.flush()

    async def _handle_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            return {
                'jsonrpc': '2.0',
                'id': None,
                'error': {
                    'code': -32700,
                    'message': f'Parse error: {exc}',
                },
            }

        request_id = request.get('id')
        method = str(request.get('method', ''))
        params = request.get('params') or {}

        if not method:
            return self._error(request_id, -32600, 'Invalid request: missing method')

        try:
            result = await self._dispatch_legacy(method=method, params=params)
            if request_id is None:
                return None
            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'result': result,
            }
        except Exception as exc:
            return self._error(request_id, -32000, str(exc))

    async def _dispatch_legacy(self, *, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == 'initialize':
            return {
                'protocolVersion': '2025-06-18',
                'capabilities': {'tools': {}},
                'serverInfo': {
                    'name': self._server_info.name,
                    'version': self._server_info.version,
                },
            }

        if method == 'ping':
            return {'ok': True}

        if method == 'tools/list':
            return {
                'tools': [
                    {
                        'name': name,
                        'description': f"Tool '{name}' exposed by MCP-RLM server",
                        'inputSchema': {'type': 'object'},
                    }
                    for name in self._registry.list_objects()
                ]
            }

        if method == 'tools/call':
            tool_name = str(params.get('name', ''))
            arguments = params.get('arguments') or {}
            context = params.get('context') or {}

            if not tool_name:
                raise RuntimeError('tools/call requires non-empty tool name')

            handler = self._registry.get(tool_name)
            invocation_ctx = MCPInvocationContext(
                episode_id=str(context.get('episode_id', '')),
                group_id=str(context.get('group_id', '')),
            )

            raw = handler(arguments, invocation_ctx)
            if inspect.isawaitable(raw):
                output = await raw
            else:
                output = raw

            return {
                'isError': False,
                'result': output,
                'content': [
                    {
                        'type': 'text',
                        'text': json.dumps(self._jsonable(output), ensure_ascii=False),
                    }
                ],
            }

        raise RuntimeError(f'Unsupported MCP method: {method}')

    async def _handle_list_tools(
        self,
        _: ServerRequestContext,
        __: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        tools = []
        for name in self._registry.list_objects():
            tools.append(
                types.Tool(
                    name=name,
                    description=f"Tool '{name}' exposed by MCP-RLM server",
                    input_schema={
                        'type': 'object',
                        'additionalProperties': True,
                    },
                )
            )
        return types.ListToolsResult(tools=tools)

    async def _handle_call_tool(
        self,
        ctx: ServerRequestContext,
        params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        name = str(params.name or '').strip()
        if not name:
            return self._error_result('tools/call requires non-empty tool name')

        arguments: Dict[str, Any] = dict(params.arguments or {})
        if not isinstance(arguments, dict):
            arguments = {}

        meta = dict(ctx.meta or {})
        inline_ctx = arguments.pop('_mcp_rlm_context', None)
        if not isinstance(inline_ctx, dict):
            inline_ctx = {}

        invocation_ctx = MCPInvocationContext(
            episode_id=str(meta.get('episode_id') or inline_ctx.get('episode_id') or ''),
            group_id=str(meta.get('group_id') or inline_ctx.get('group_id') or ''),
        )

        try:
            handler = self._registry.get(name)
            raw = handler(arguments, invocation_ctx)
            if inspect.isawaitable(raw):
                output = await raw
            else:
                output = raw

            rendered = json.dumps(self._jsonable(output), ensure_ascii=False)
            structured_content = self._jsonable(output) if isinstance(output, dict) else None
            return types.CallToolResult(
                content=[types.TextContent(type='text', text=rendered)],
                structured_content=structured_content,
                is_error=False,
            )
        except Exception as exc:
            return self._error_result(str(exc))

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [StdioMCPServer._jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): StdioMCPServer._jsonable(v) for k, v in value.items()}
        return str(value)

    @staticmethod
    def _error_result(message: str) -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type='text', text=message)],
            structured_content={'error': message},
            is_error=True,
        )

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            'jsonrpc': '2.0',
            'id': request_id,
            'error': {
                'code': code,
                'message': message,
            },
        }
