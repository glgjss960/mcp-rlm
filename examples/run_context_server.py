from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import argparse
import asyncio
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_rlm import MCPRegistry, MCPServerInfo, StdioMCPServer
from mcp_rlm.mvp import LongContextStoreCache


def _resolve_manifest(params: Dict[str, Any], default_manifest: str | None) -> str:
    manifest = str(params.get("manifest_path") or "").strip()
    if manifest:
        return manifest
    if default_manifest:
        return default_manifest
    raise RuntimeError("manifest_path is required for context tools")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run context MCP server")
    parser.add_argument("--manifest", type=str, default="", help="Default manifest path")
    args = parser.parse_args()

    default_manifest = args.manifest.strip() or os.getenv("MCP_RLM_CONTEXT_MANIFEST", "").strip() or None
    cache = LongContextStoreCache()

    registry = MCPRegistry()

    def context_stats(payload: Dict[str, Any], _) -> Dict[str, Any]:
        manifest = _resolve_manifest(payload, default_manifest)
        store = cache.get(manifest)
        return store.context_stats()

    def list_level(payload: Dict[str, Any], _) -> Dict[str, Any]:
        manifest = _resolve_manifest(payload, default_manifest)
        store = cache.get(manifest)
        level = int(payload.get("level", 0))
        limit = int(payload.get("limit", 100))
        offset = int(payload.get("offset", 0))
        return store.list_level(level, limit=limit, offset=offset)

    def read_segment(payload: Dict[str, Any], _) -> Dict[str, Any]:
        manifest = _resolve_manifest(payload, default_manifest)
        store = cache.get(manifest)
        segment_id = str(payload.get("segment_id", "")).strip()
        if not segment_id:
            raise RuntimeError("read_segment requires segment_id")
        max_chars = payload.get("max_chars")
        max_chars_int = None if max_chars is None else int(max_chars)
        offset = int(payload.get("offset", 0))
        return store.read_segment(segment_id, max_chars=max_chars_int, offset=offset)

    def search_hierarchical(payload: Dict[str, Any], _) -> Dict[str, Any]:
        manifest = _resolve_manifest(payload, default_manifest)
        store = cache.get(manifest)
        query = str(payload.get("query", "")).strip()
        if not query:
            raise RuntimeError("search_hierarchical requires query")
        top_k = int(payload.get("top_k", 12))
        coarse_k = int(payload.get("coarse_k", 24))
        return store.search_hierarchical(query=query, top_k=top_k, coarse_k=coarse_k)

    registry.register("context_stats", context_stats)
    registry.register("list_level", list_level)
    registry.register("read_segment", read_segment)
    registry.register("search_hierarchical", search_hierarchical)

    server = StdioMCPServer(
        registry=registry,
        server_info=MCPServerInfo(name="mcp-rlm-context-server", version="0.1.0"),
    )
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
