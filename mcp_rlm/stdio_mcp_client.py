from __future__ import annotations

from contextlib import AsyncExitStack
from datetime import timedelta
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence
import asyncio
import contextlib
import json

from .mcp import MCPCall, MCPInvocationContext, MCPResult
from .mcp_sdk import ensure_mcp_sdk, mcp_sdk_status

_SDK_AVAILABLE = False
_SDK_IMPORT_ERROR: Optional[Exception] = None

try:
    ensure_mcp_sdk()
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _SDK_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    ClientSession = Any  # type: ignore[misc,assignment]
    StdioServerParameters = Any  # type: ignore[misc,assignment]
    stdio_client = None  # type: ignore[assignment]
    _SDK_IMPORT_ERROR = exc


class StdioMCPClient:
    """Remote MCP tool client over stdio.

    Uses official MCP Python SDK when available; otherwise falls back to legacy JSON-RPC transport.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        max_concurrency: int = 32,
        startup_timeout_seconds: float = 10.0,
        default_request_timeout_seconds: float = 30.0,
        prefer_official_sdk: bool = True,
        strict_official_sdk: bool = False,
    ) -> None:
        if not command:
            raise ValueError('command must not be empty')
        if strict_official_sdk and not prefer_official_sdk:
            raise ValueError('strict_official_sdk=True requires prefer_official_sdk=True')

        self._command = [str(x) for x in command]
        self._cwd = cwd
        self._env = env
        self._startup_timeout_seconds = startup_timeout_seconds
        self._default_request_timeout_seconds = default_request_timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)

        self._use_sdk = bool(prefer_official_sdk and _SDK_AVAILABLE)

        if strict_official_sdk and not self._use_sdk:
            ok, detail = mcp_sdk_status()
            reason = detail if not ok else 'MCP SDK is available but not selected'
            if _SDK_IMPORT_ERROR is not None:
                reason = f'{reason}; import_error={type(_SDK_IMPORT_ERROR).__name__}: {_SDK_IMPORT_ERROR}'
            raise RuntimeError(f'Official MCP SDK is required but unavailable. {reason}')

        self._strict_official_sdk = bool(strict_official_sdk)
        self._start_lock = asyncio.Lock()

        # SDK-mode state
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

        # Legacy-mode state
        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._request_lock = asyncio.Lock()
        self._next_id = 1
        self._pending: Dict[int, asyncio.Future[Any]] = {}
        self._stderr_buffer: List[str] = []
        self._started = False

    @property
    def using_official_sdk(self) -> bool:
        return self._use_sdk

    async def start(self) -> None:
        if self._use_sdk:
            try:
                await self._start_sdk()
                return
            except Exception:
                if self._strict_official_sdk:
                    raise
                # SDK startup can fail when server and client support mismatched protocol versions.
                # Fall back to legacy JSON-RPC transport to keep eval/inference running.
                self._use_sdk = False
        await self._start_legacy()

    async def close(self) -> None:
        if self._use_sdk:
            await self._close_sdk()
        else:
            await self._close_legacy()

    async def __aenter__(self) -> 'StdioMCPClient':
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def list_tools(self) -> List[str]:
        await self.start()
        if self._use_sdk:
            session = self._require_session()
            response = await session.list_tools()
            return [str(tool.name) for tool in response.tools if getattr(tool, 'name', None)]

        result = await self._send_request('tools/list', {}, timeout_seconds=self._default_request_timeout_seconds)
        tools = result.get('tools', []) if isinstance(result, dict) else []
        return [str(t.get('name')) for t in tools if isinstance(t, dict) and t.get('name')]

    async def call(self, call: MCPCall, ctx: MCPInvocationContext) -> MCPResult:
        if self._use_sdk:
            return await self._call_sdk(call, ctx)
        return await self._call_legacy(call, ctx)

    async def call_many(self, calls: List[MCPCall], ctx: MCPInvocationContext) -> List[MCPResult]:
        tasks = [asyncio.create_task(self.call(call, ctx)) for call in calls]
        return await asyncio.gather(*tasks)

    async def _start_sdk(self) -> None:
        async with self._start_lock:
            if self._session is not None:
                return

            stack = AsyncExitStack()
            try:
                command = self._command[0]
                args = self._command[1:]
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    cwd=self._cwd,
                    env=self._env,
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await asyncio.wait_for(session.initialize(), timeout=self._startup_timeout_seconds)
            except Exception:
                await stack.aclose()
                raise

            self._exit_stack = stack
            self._session = session

    async def _close_sdk(self) -> None:
        async with self._start_lock:
            stack = self._exit_stack
            self._exit_stack = None
            self._session = None

        if stack is not None:
            await stack.aclose()

    def _require_session(self) -> ClientSession:
        session = self._session
        if session is None:
            raise RuntimeError('MCP session is not started')
        return session

    async def _call_sdk(self, call: MCPCall, ctx: MCPInvocationContext) -> MCPResult:
        start = perf_counter()
        try:
            await self.start()
            session = self._require_session()

            payload = dict(call.payload or {})
            inline_ctx = payload.get('_mcp_rlm_context')
            if not isinstance(inline_ctx, dict):
                payload['_mcp_rlm_context'] = {
                    'episode_id': ctx.episode_id,
                    'group_id': ctx.group_id,
                }

            timeout_delta = timedelta(seconds=max(0.001, float(call.timeout_seconds)))

            async with self._semaphore:
                result = await session.call_tool(
                    name=call.object_name,
                    arguments=payload,
                    read_timeout_seconds=timeout_delta,
                    meta={
                        'episode_id': ctx.episode_id,
                        'group_id': ctx.group_id,
                    },
                )

            if self._tool_result_is_error(result):
                raise RuntimeError(self._extract_error(result))

            output = self._tool_result_structured_content(result)
            if output is None:
                output = self._extract_output_from_content(result.content)

            return MCPResult(
                object_name=call.object_name,
                ok=True,
                output=output,
                latency_ms=int((perf_counter() - start) * 1000),
            )
        except Exception as exc:
            return MCPResult(
                object_name=call.object_name,
                ok=False,
                error=str(exc),
                latency_ms=int((perf_counter() - start) * 1000),
            )

    async def _start_legacy(self) -> None:
        async with self._start_lock:
            if self._started:
                return

            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                cwd=self._cwd,
                env=self._env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            if self._process.stdin is None or self._process.stdout is None or self._process.stderr is None:
                raise RuntimeError('Failed to create stdio pipes for MCP server process')

            self._reader_task = asyncio.create_task(self._reader_loop())
            self._stderr_task = asyncio.create_task(self._stderr_loop())

            await self._send_request(
                method='initialize',
                params={'clientInfo': {'name': 'mcp-rlm-client', 'version': '0.1.0'}},
                timeout_seconds=self._startup_timeout_seconds,
            )
            self._started = True

    async def _close_legacy(self) -> None:
        process = self._process
        if process is None:
            return

        reader_task = self._reader_task
        stderr_task = self._stderr_task
        task_list = [t for t in [reader_task, stderr_task] if t is not None]

        for task in task_list:
            task.cancel()

        if process.stdin is not None and not process.stdin.is_closing():
            process.stdin.close()
            with contextlib.suppress(Exception):
                await process.stdin.wait_closed()

        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

        if task_list:
            await asyncio.gather(*task_list, return_exceptions=True)

        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(RuntimeError('MCP client closed'))
        self._pending.clear()

        self._process = None
        self._reader_task = None
        self._stderr_task = None
        self._started = False

    async def _call_legacy(self, call: MCPCall, ctx: MCPInvocationContext) -> MCPResult:
        start = perf_counter()
        try:
            await self.start()
            async with self._semaphore:
                response = await self._send_request(
                    method='tools/call',
                    params={
                        'name': call.object_name,
                        'arguments': call.payload,
                        'context': {
                            'episode_id': ctx.episode_id,
                            'group_id': ctx.group_id,
                        },
                    },
                    timeout_seconds=call.timeout_seconds,
                )

            if not isinstance(response, dict):
                raise RuntimeError('Invalid tools/call response')
            if response.get('isError'):
                raise RuntimeError(str(response.get('error', 'MCP tool error')))

            output = response.get('result')
            if output is None:
                output = self._extract_output_from_content(response.get('content'))

            return MCPResult(
                object_name=call.object_name,
                ok=True,
                output=output,
                latency_ms=int((perf_counter() - start) * 1000),
            )
        except Exception as exc:
            return MCPResult(
                object_name=call.object_name,
                ok=False,
                error=str(exc),
                latency_ms=int((perf_counter() - start) * 1000),
            )

    async def _send_request(self, method: str, params: Dict[str, Any], *, timeout_seconds: float) -> Any:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError('MCP server process not started')

        request_id = self._next_request_id()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future

        message = {
            'jsonrpc': '2.0',
            'id': request_id,
            'method': method,
            'params': params,
        }
        payload = (json.dumps(message, ensure_ascii=False) + '\n').encode('utf-8')

        async with self._request_lock:
            self._process.stdin.write(payload)
            await self._process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except Exception:
            self._pending.pop(request_id, None)
            raise

    def _next_request_id(self) -> int:
        request_id = self._next_id
        self._next_id += 1
        return request_id

    async def _reader_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while True:
                raw = await self._process.stdout.readline()
                if not raw:
                    break

                line = raw.decode('utf-8').strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue

                request_id = message.get('id')
                if request_id is None:
                    continue

                try:
                    numeric_id = int(request_id)
                except (TypeError, ValueError):
                    continue

                future = self._pending.pop(numeric_id, None)
                if future is None or future.done():
                    continue

                if 'error' in message:
                    error_obj = message.get('error') or {}
                    code = error_obj.get('code', -32000)
                    msg = error_obj.get('message', 'Unknown JSON-RPC error')
                    future.set_exception(RuntimeError(f'MCP JSON-RPC error {code}: {msg}'))
                else:
                    future.set_result(message.get('result'))
        finally:
            err_suffix = ''
            if self._stderr_buffer:
                err_suffix = ' | stderr: ' + '\n'.join(self._stderr_buffer[-8:])
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(RuntimeError('MCP server stream closed' + err_suffix))
            self._pending.clear()

    async def _stderr_loop(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        while True:
            raw = await self._process.stderr.readline()
            if not raw:
                break
            line = raw.decode('utf-8', errors='replace').rstrip()
            if line:
                self._stderr_buffer.append(line)
                if len(self._stderr_buffer) > 200:
                    self._stderr_buffer = self._stderr_buffer[-200:]

    @staticmethod
    def _extract_error(result: Any) -> str:
        structured = StdioMCPClient._tool_result_structured_content(result)
        if isinstance(structured, dict) and structured.get('error'):
            return str(structured.get('error'))
        extracted = StdioMCPClient._extract_output_from_content(getattr(result, 'content', None))
        if extracted is None:
            return 'MCP tool error'
        return str(extracted)

    @staticmethod
    def _tool_result_is_error(result: Any) -> bool:
        raw = getattr(result, 'isError', None)
        if raw is None:
            raw = getattr(result, 'is_error', None)
        return bool(raw)

    @staticmethod
    def _tool_result_structured_content(result: Any) -> Any:
        raw = getattr(result, 'structuredContent', None)
        if raw is None:
            raw = getattr(result, 'structured_content', None)
        return raw

    @staticmethod
    def _extract_output_from_content(content: Any) -> Any:
        if not isinstance(content, list):
            return None
        for item in content:
            text: Optional[str] = None
            if isinstance(item, dict):
                maybe_text = item.get('text')
                if isinstance(maybe_text, str):
                    text = maybe_text
            else:
                maybe_text = getattr(item, 'text', None)
                if isinstance(maybe_text, str):
                    text = maybe_text
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return None

