from __future__ import annotations

from pathlib import Path
import argparse
import asyncio
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_rlm import MCPRegistry, MCPServerInfo, StdioMCPServer, register_builtin_objects


async def main() -> None:
    parser = argparse.ArgumentParser(description='Run analysis MCP server')
    parser.add_argument('--legacy-mcp', action='store_true', help='Use legacy JSON-RPC transport instead of official MCP SDK')
    parser.add_argument('--require-official-sdk', action='store_true', help='Fail fast if official MCP SDK cannot be used')
    args = parser.parse_args()

    registry = MCPRegistry()
    register_builtin_objects(registry)

    server = StdioMCPServer(
        registry=registry,
        server_info=MCPServerInfo(name='mcp-rlm-analysis-server', version='0.1.0'),
        prefer_official_sdk=not args.legacy_mcp,
        strict_official_sdk=bool(args.require_official_sdk),
    )
    await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main())
