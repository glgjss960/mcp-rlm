from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from mcp_rlm import (
    MCPClient,
    MCPRegistry,
    MCPRLMRuntime,
    ProgramRegistry,
    SharedMemory,
    register_builtin_objects,
    register_builtin_programs,
)
from mcp_rlm.training import export_agentic_rl, export_cold_start, export_trace, export_verl, export_openrlhf


class TestExporters(unittest.TestCase):
    def test_export_files(self) -> None:
        async def runner():
            object_registry = MCPRegistry()
            register_builtin_objects(object_registry)

            program_registry = ProgramRegistry()
            register_builtin_programs(program_registry)

            runtime = MCPRLMRuntime(
                program_registry=program_registry,
                mcp_client=MCPClient(object_registry),
                memory=SharedMemory(),
            )

            return await runtime.run_episode(
                goal="what is the code",
                program="root_map_reduce",
                input_payload={
                    "query": "what is the secret code",
                    "documents": [
                        "noise doc",
                        "secret code is 555-A10",
                        "another noise doc",
                    ],
                    "chunk_size": 1,
                },
            )

        trace = asyncio.run(runner())

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export_trace(trace, out)
            export_cold_start([trace], out / "cold_start_turns.jsonl")
            export_agentic_rl([trace], out / "agentic_rl.jsonl")
            export_verl([trace], out / "verl_warm_start.jsonl")
            export_openrlhf([trace], out / "openrlhf_episodes.jsonl")

            required = [
                out / "episodes.jsonl",
                out / "groups.jsonl",
                out / "steps.jsonl",
                out / "memory_events.jsonl",
                out / "cold_start_turns.jsonl",
                out / "agentic_rl.jsonl",
                out / "verl_warm_start.jsonl",
                out / "openrlhf_episodes.jsonl",
            ]
            for path in required:
                self.assertTrue(path.exists(), str(path))
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
