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
    preprocess_long_context,
    register_builtin_programs,
    register_mvp_programs,
)
from mcp_rlm.training import export_agentic_rl, export_cold_start, export_openrlhf, export_trace, export_verl


async def main() -> None:
    parser = argparse.ArgumentParser(description="One-command MVP pipeline: preprocess + inference")
    parser.add_argument("--input", type=str, required=True, help="Path to raw long text")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--store", type=str, default="artifacts/context_store")
    parser.add_argument("--out", type=str, default="artifacts/mvp_pipeline")
    parser.add_argument("--chunk-chars", type=int, default=16000)
    parser.add_argument("--overlap-chars", type=int, default=400)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument("--max-children", type=int, default=16)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    store_dir = (ROOT / args.store).resolve()
    manifest_path = preprocess_long_context(
        input_file=input_path,
        output_dir=store_dir,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        branch_factor=args.branch_factor,
    )

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx_server = ROOT / "examples" / "run_context_server.py"
    analysis_server = ROOT / "examples" / "run_analysis_server.py"

    mcp_client = MultiServerMCPClient(
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
        mcp_client=mcp_client,
        memory=FileSharedMemory(out_dir / "memory"),
        max_group_concurrency=64,
    )

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

        print("Manifest:", manifest_path)
        print("Episode:", trace.episode_id)
        print("Success:", trace.success)
        print("Root output:")
        print(json.dumps(trace.root_output, ensure_ascii=False, indent=2))
        print("Output directory:", out_dir)
    finally:
        await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
