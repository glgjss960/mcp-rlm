from __future__ import annotations

from pathlib import Path
import argparse
import asyncio
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_rlm import (
    FileSharedMemory,
    MCPRLMRuntime,
    MCPServerSpec,
    MultiServerMCPClient,
    ProgramRegistry,
    register_builtin_programs,
    register_mvp_programs,
)
from mcp_rlm.training import export_agentic_rl, export_cold_start, export_openrlhf, export_trace, export_verl


def build_runtime(manifest_path: Path, memory_dir: Path) -> tuple[MCPRLMRuntime, MultiServerMCPClient]:
    ctx_server = ROOT / "examples" / "run_context_server.py"
    analysis_server = ROOT / "examples" / "run_analysis_server.py"

    multi_client = MultiServerMCPClient(
        specs=[
            MCPServerSpec(
                alias="ctx",
                command=[sys.executable, str(ctx_server), "--manifest", str(manifest_path)],
                cwd=str(ROOT),
                max_concurrency=24,
            ),
            MCPServerSpec(
                alias="analysis",
                command=[sys.executable, str(analysis_server)],
                cwd=str(ROOT),
                max_concurrency=24,
            ),
        ]
    )

    registry = ProgramRegistry()
    register_builtin_programs(registry)
    register_mvp_programs(registry)

    runtime = MCPRLMRuntime(
        program_registry=registry,
        mcp_client=multi_client,
        memory=FileSharedMemory(memory_dir),
        max_group_concurrency=64,
    )
    return runtime, multi_client


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCP-RLM MVP inference with multi-server MCP orchestration")
    parser.add_argument("--manifest", type=str, required=True, help="Path to context manifest.json")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--out", type=str, default="artifacts/mvp")
    parser.add_argument("--max-children", type=int, default=16)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = out_dir / "memory"

    runtime, multi_client = build_runtime(manifest_path, memory_dir)

    try:
        trace = await runtime.run_episode(
            goal=args.query,
            program="mvp_root",
            input_payload={
                "query": args.query,
                "manifest_path": str(manifest_path),
                "max_children": args.max_children,
            },
        )

        export_trace(trace, out_dir)
        export_cold_start([trace], out_dir / "cold_start_turns.jsonl")
        export_agentic_rl([trace], out_dir / "agentic_rl.jsonl")
        export_verl([trace], out_dir / "verl_warm_start.jsonl")
        export_openrlhf([trace], out_dir / "openrlhf_episodes.jsonl")

        print("Episode:", trace.episode_id)
        print("Success:", trace.success)
        print("Root output:")
        print(json.dumps(trace.root_output, ensure_ascii=False, indent=2))
        print("Output directory:", out_dir)
        print("Memory directory:", memory_dir)
    finally:
        await multi_client.close()


if __name__ == "__main__":
    asyncio.run(main())
