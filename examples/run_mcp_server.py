from __future__ import annotations

from pathlib import Path
import asyncio
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_rlm import MCPRegistry, MCPServerInfo, StdioMCPServer, register_builtin_objects


async def main() -> None:
    registry = MCPRegistry()
    register_builtin_objects(registry)

    server = StdioMCPServer(
        registry=registry,
        server_info=MCPServerInfo(name="mcp-rlm-builtin-server", version="0.1.0"),
    )
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
