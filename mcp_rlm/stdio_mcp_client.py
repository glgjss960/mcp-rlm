from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence
import asyncio
import contextlib
import json

from .mcp import MCPCall, MCPInvocationContext, MCPResult


class StdioMCPClient:
    """Remote MCP tool client over stdio JSON-RPC, exposing MCPClient-compatible call APIs."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        max_concurrency: int = 32,
        startup_timeout_seconds: float = 10.0,
        default_request_timeout_seconds: float = 30.0,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")

        self._command = list(command)
        self._cwd = cwd
        self._env = env
        self._startup_timeout_seconds = startup_timeout_seconds
        self._default_request_timeout_seconds = default_request_timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)

        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._start_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._next_id = 1
        self._pending: Dict[int, asyncio.Future[Any]] = {}
        self._stderr_buffer: List[str] = []
        self._started = False

    async def start(self) -> None:
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
                raise RuntimeError("Failed to create stdio pipes for MCP server process")

            self._reader_task = asyncio.create_task(self._reader_loop())
            self._stderr_task = asyncio.create_task(self._stderr_loop())

            await self._send_request(
                method="initialize",
                params={"clientInfo": {"name": "mcp-rlm-client", "version": "0.1.0"}},
                timeout_seconds=self._startup_timeout_seconds,
            )
            self._started = True

    async def close(self) -> None:
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
                future.set_exception(RuntimeError("MCP client closed"))
        self._pending.clear()

        self._process = None
        self._reader_task = None
        self._stderr_task = None
        self._started = False

    async def __aenter__(self) -> "StdioMCPClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def list_tools(self) -> List[str]:
        await self.start()
        result = await self._send_request("tools/list", {}, timeout_seconds=self._default_request_timeout_seconds)
        tools = result.get("tools", []) if isinstance(result, dict) else []
        return [str(t.get("name")) for t in tools if isinstance(t, dict) and t.get("name")]

    async def call(self, call: MCPCall, ctx: MCPInvocationContext) -> MCPResult:
        start = perf_counter()
        try:
            await self.start()
            async with self._semaphore:
                response = await self._send_request(
                    method="tools/call",
                    params={
                        "name": call.object_name,
                        "arguments": call.payload,
                        "context": {
                            "episode_id": ctx.episode_id,
                            "group_id": ctx.group_id,
                        },
                    },
                    timeout_seconds=call.timeout_seconds,
                )

            if not isinstance(response, dict):
                raise RuntimeError("Invalid tools/call response")
            if response.get("isError"):
                raise RuntimeError(str(response.get("error", "MCP tool error")))

            output = response.get("result")
            if output is None:
                output = self._extract_output_from_content(response.get("content"))

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

    async def call_many(self, calls: List[MCPCall], ctx: MCPInvocationContext) -> List[MCPResult]:
        tasks = [asyncio.create_task(self.call(call, ctx)) for call in calls]
        return await asyncio.gather(*tasks)

    async def _send_request(self, method: str, params: Dict[str, Any], *, timeout_seconds: float) -> Any:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("MCP server process not started")

        request_id = self._next_request_id()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        payload = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")

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

                line = raw.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue

                request_id = message.get("id")
                if request_id is None:
                    continue

                try:
                    numeric_id = int(request_id)
                except (TypeError, ValueError):
                    continue

                future = self._pending.pop(numeric_id, None)
                if future is None or future.done():
                    continue

                if "error" in message:
                    error_obj = message.get("error") or {}
                    code = error_obj.get("code", -32000)
                    msg = error_obj.get("message", "Unknown JSON-RPC error")
                    future.set_exception(RuntimeError(f"MCP JSON-RPC error {code}: {msg}"))
                else:
                    future.set_result(message.get("result"))
        finally:
            err_suffix = ""
            if self._stderr_buffer:
                err_suffix = " | stderr: " + "\n".join(self._stderr_buffer[-8:])
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(RuntimeError("MCP server stream closed" + err_suffix))
            self._pending.clear()

    async def _stderr_loop(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        while True:
            raw = await self._process.stderr.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                self._stderr_buffer.append(line)
                if len(self._stderr_buffer) > 200:
                    self._stderr_buffer = self._stderr_buffer[-200:]

    @staticmethod
    def _extract_output_from_content(content: Any) -> Any:
        if not isinstance(content, list):
            return None
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return None
