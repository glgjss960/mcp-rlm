from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

from mcp_rlm import MCPCall, MCPInvocationContext, MCPRLMRuntime, ProgramRegistry, SharedMemory, StdioMCPClient, register_builtin_programs


class TestStdioMCP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        server_script = repo_root / "examples" / "run_mcp_server.py"

        self.client = StdioMCPClient(
            command=[sys.executable, str(server_script)],
            cwd=str(repo_root),
            max_concurrency=16,
        )
        await self.client.start()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_list_and_call_tools(self) -> None:
        tools = await self.client.list_tools()
        self.assertIn("extract_facts", tools)
        self.assertIn("merge_facts", tools)

        result = await self.client.call(
            MCPCall(
                object_name="extract_facts",
                payload={
                    "query": "what is secret code",
                    "documents": ["The secret code is 123-A77."],
                    "max_hits": 3,
                },
            ),
            MCPInvocationContext(episode_id="ep_test", group_id="grp_test"),
        )
        self.assertTrue(result.ok)
        self.assertGreaterEqual(result.output.get("total_hits", 0), 1)

    async def test_runtime_with_remote_server(self) -> None:
        program_registry = ProgramRegistry()
        register_builtin_programs(program_registry)

        runtime = MCPRLMRuntime(
            program_registry=program_registry,
            mcp_client=self.client,
            memory=SharedMemory(),
        )

        trace = await runtime.run_episode(
            goal="Find secret code",
            program="root_map_reduce",
            input_payload={
                "query": "What is the secret code?",
                "documents": [
                    "notes",
                    "Confidential: the secret code is 888-C22.",
                    "more notes",
                ],
                "chunk_size": 1,
            },
        )
        self.assertTrue(trace.success)
        self.assertIsInstance(trace.root_output, dict)


if __name__ == "__main__":
    unittest.main()
