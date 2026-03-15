from __future__ import annotations

from pathlib import Path
import asyncio
import tempfile
import unittest
import sys

from mcp_rlm import (
    FileSharedMemory,
    MCPRLMRuntime,
    MCPServerSpec,
    MultiServerMCPClient,
    ProgramRegistry,
    preprocess_long_context,
    register_builtin_programs,
    register_mvp_programs,
)


class TestMVPInference(unittest.IsolatedAsyncioTestCase):
    async def test_mvp_end_to_end(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        ctx_server = repo_root / "examples" / "run_context_server.py"
        analysis_server = repo_root / "examples" / "run_analysis_server.py"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_file = tmp_path / "long.txt"

            blocks = [f"section {i}: random text and logs." for i in range(500)]
            blocks.insert(233, "final truth: the launch code is 123-B77 and should be reported exactly.")
            input_file.write_text("\n".join(blocks), encoding="utf-8")

            manifest = preprocess_long_context(
                input_file=input_file,
                output_dir=tmp_path / "store",
                chunk_chars=1000,
                overlap_chars=120,
                branch_factor=4,
            )

            client = MultiServerMCPClient(
                specs=[
                    MCPServerSpec(
                        alias="ctx",
                        command=[sys.executable, str(ctx_server), "--manifest", str(manifest)],
                        cwd=str(repo_root),
                    ),
                    MCPServerSpec(
                        alias="analysis",
                        command=[sys.executable, str(analysis_server)],
                        cwd=str(repo_root),
                    ),
                ]
            )

            registry = ProgramRegistry()
            register_builtin_programs(registry)
            register_mvp_programs(registry)

            runtime = MCPRLMRuntime(
                program_registry=registry,
                mcp_client=client,
                memory=FileSharedMemory(tmp_path / "memory"),
                max_group_concurrency=32,
            )

            try:
                trace = await runtime.run_episode(
                    goal="What is the launch code?",
                    program="mvp_root",
                    input_payload={
                        "query": "What is the launch code?",
                        "manifest_path": str(manifest),
                        "max_children": 8,
                    },
                )
            finally:
                await client.close()

            self.assertTrue(trace.success)
            self.assertIsInstance(trace.root_output, dict)
            output = trace.root_output if isinstance(trace.root_output, dict) else {}
            self.assertIn("answer", output)
            self.assertIn("123-B77", str(output.get("answer", "")))


if __name__ == "__main__":
    unittest.main()
