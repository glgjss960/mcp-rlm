from __future__ import annotations

from pathlib import Path
import shutil
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
    register_longbench_v2_programs,
    register_mvp_programs,
)


class TestLongBenchV2Inference(unittest.IsolatedAsyncioTestCase):
    async def test_longbench_v2_root_predicts_option(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        ctx_server = repo_root / "examples" / "run_context_server.py"
        analysis_server = repo_root / "examples" / "run_analysis_server.py"

        tmp_base = repo_root / "artifacts" / "tmp_tests"
        tmp_base.mkdir(parents=True, exist_ok=True)
        tmp = tmp_base / "lb2_case"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)

        try:
            input_file = tmp / "long.txt"

            filler = [f"paragraph {i}: historical random note." for i in range(320)]
            filler.insert(160, "Key fact: The capital of France is Paris, and this is definitive.")
            input_file.write_text("\n".join(filler), encoding="utf-8")

            manifest = preprocess_long_context(
                input_file=input_file,
                output_dir=tmp / "store",
                chunk_chars=1200,
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
            register_longbench_v2_programs(registry)

            runtime = MCPRLMRuntime(
                program_registry=registry,
                mcp_client=client,
                memory=FileSharedMemory(tmp / "memory"),
                max_group_concurrency=32,
            )

            try:
                trace = await runtime.run_episode(
                    goal="Which city is the capital of France?",
                    program="longbench_v2_root",
                    input_payload={
                        "question": "Which city is the capital of France?",
                        "choices": {
                            "A": "Berlin",
                            "B": "Madrid",
                            "C": "Paris",
                            "D": "Rome",
                        },
                        "manifest_path": str(manifest),
                        "max_children": 8,
                    },
                )
            finally:
                await client.close()

            self.assertTrue(trace.success)
            self.assertIsInstance(trace.root_output, dict)
            output = trace.root_output if isinstance(trace.root_output, dict) else {}
            self.assertEqual(output.get("pred"), "C")
            self.assertIn("(C)", str(output.get("response", "")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
