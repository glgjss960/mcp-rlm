from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from time import perf_counter

from .mcp import MCPCall, MCPInvocationContext, MCPResult
from .stdio_mcp_client import StdioMCPClient


def _to_bool(raw: Any, *, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_timeout(raw: Any, *, default: float, low: float = 0.1, high: float = 7200.0) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _stage_log(
    *,
    enabled: bool,
    component: str,
    stage: str,
    status: str,
    elapsed_ms: Optional[int] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    if not enabled:
        return
    payload: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "stage": stage,
        "status": status,
    }
    if elapsed_ms is not None:
        payload["elapsed_ms"] = int(elapsed_ms)
    if detail:
        payload["detail"] = detail
    print("[mcp-rlm-stage] " + json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


@dataclass
class MCPServerSpec:
    alias: str
    command: Sequence[str]
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    max_concurrency: int = 32
    prefer_official_sdk: bool = True
    strict_official_sdk: bool = False


class MultiServerMCPClient:
    """Route MCP tool calls across multiple stdio MCP servers.

    Object names use the format: '<alias>/<tool_name>'.
    """

    def __init__(self, specs: List[MCPServerSpec], *, default_alias: Optional[str] = None) -> None:
        if not specs:
            raise ValueError('specs must not be empty')

        self._clients: Dict[str, StdioMCPClient] = {}
        for spec in specs:
            if spec.alias in self._clients:
                raise ValueError(f'Duplicate alias: {spec.alias}')
            self._clients[spec.alias] = StdioMCPClient(
                command=list(spec.command),
                cwd=spec.cwd,
                env=spec.env,
                max_concurrency=spec.max_concurrency,
                prefer_official_sdk=spec.prefer_official_sdk,
                strict_official_sdk=spec.strict_official_sdk,
            )

        self.default_alias = default_alias
        if self.default_alias and self.default_alias not in self._clients:
            raise ValueError(f'default_alias not found in specs: {self.default_alias}')

        self._stage_logs_enabled = _to_bool(os.getenv("MCP_RLM_DEBUG_STAGE_LOGS"), default=True)
        self._default_list_timeout_seconds = _to_timeout(
            os.getenv("MCP_RLM_MANAGER_LIST_OBJECTS_TIMEOUT_SECONDS", "20"),
            default=20.0,
        )

    async def start(self) -> None:
        await asyncio.gather(*(client.start() for client in self._clients.values()))

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self._clients.values()), return_exceptions=True)

    async def __aenter__(self) -> 'MultiServerMCPClient':
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def list_objects(self, *, timeout_seconds: Optional[float] = None) -> List[str]:
        resolved_timeout = _to_timeout(
            timeout_seconds if timeout_seconds is not None else self._default_list_timeout_seconds,
            default=self._default_list_timeout_seconds,
        )
        start = perf_counter()
        _stage_log(
            enabled=self._stage_logs_enabled,
            component="mcp_client.multi",
            stage="list_objects",
            status="start",
            detail={"num_servers": len(self._clients), "timeout_seconds": resolved_timeout},
        )

        async def one(alias: str, client: StdioMCPClient) -> List[str]:
            one_start = perf_counter()
            _stage_log(
                enabled=self._stage_logs_enabled,
                component="mcp_client.multi",
                stage="list_tools",
                status="start",
                detail={"alias": alias, "timeout_seconds": resolved_timeout},
            )
            try:
                tools = await client.list_tools(timeout_seconds=resolved_timeout)
            except Exception as exc:
                _stage_log(
                    enabled=self._stage_logs_enabled,
                    component="mcp_client.multi",
                    stage="list_tools",
                    status="error",
                    elapsed_ms=int((perf_counter() - one_start) * 1000),
                    detail={"alias": alias, "error": str(exc), "error_type": type(exc).__name__},
                )
                return []
            _stage_log(
                enabled=self._stage_logs_enabled,
                component="mcp_client.multi",
                stage="list_tools",
                status="ok",
                elapsed_ms=int((perf_counter() - one_start) * 1000),
                detail={"alias": alias, "num_tools": len(tools)},
            )
            return [f'{alias}/{tool}' for tool in tools]

        tasks = [asyncio.create_task(one(alias, client)) for alias, client in self._clients.items()]
        parts = await asyncio.gather(*tasks, return_exceptions=True)

        out: List[str] = []
        for part in parts:
            if isinstance(part, list):
                out.extend(part)

        _stage_log(
            enabled=self._stage_logs_enabled,
            component="mcp_client.multi",
            stage="list_objects",
            status="ok",
            elapsed_ms=int((perf_counter() - start) * 1000),
            detail={"num_objects": len(out)},
        )
        return sorted(out)

    async def call(self, call: MCPCall, ctx: MCPInvocationContext) -> MCPResult:
        alias, tool_name = self._route(call.object_name)
        routed = MCPCall(object_name=tool_name, payload=call.payload, timeout_seconds=call.timeout_seconds)
        return await self._clients[alias].call(routed, ctx)

    async def call_many(self, calls: List[MCPCall], ctx: MCPInvocationContext) -> List[MCPResult]:
        async def one(call: MCPCall) -> MCPResult:
            try:
                return await self.call(call, ctx)
            except asyncio.CancelledError as exc:
                return MCPResult(object_name=call.object_name, ok=False, error=f"CancelledError: {exc}")
            except Exception as exc:
                return MCPResult(object_name=call.object_name, ok=False, error=str(exc))

        tasks = [asyncio.create_task(one(call)) for call in calls]
        return await asyncio.gather(*tasks)

    def _route(self, object_name: str) -> tuple[str, str]:
        if '/' in object_name:
            alias, tool_name = object_name.split('/', 1)
            if alias not in self._clients:
                raise KeyError(f'Unknown MCP server alias: {alias}')
            if not tool_name:
                raise ValueError(f'Invalid object name: {object_name}')
            return alias, tool_name

        if self.default_alias is None:
            raise ValueError(
                f"Object name '{object_name}' has no alias and no default_alias configured. Use '<alias>/<tool>'"
            )
        return self.default_alias, object_name
