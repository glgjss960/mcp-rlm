from __future__ import annotations

from pathlib import Path
import asyncio
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def build_runtime() -> MCPRLMRuntime:
    object_registry = MCPRegistry()
    register_builtin_objects(object_registry)

    program_registry = ProgramRegistry()
    register_builtin_programs(program_registry)

    return MCPRLMRuntime(
        program_registry=program_registry,
        mcp_client=MCPClient(object_registry, max_concurrency=64),
        memory=SharedMemory(),
    )


async def main() -> None:
    runtime = build_runtime()

    docs = [
        "Meeting note: Project Atlas has three milestones. Timeline is under review.",
        "Audit memo: budget updates and staffing changes for Q4.",
        "Confidential: the secret code for project atlas is 731-B42. Keep internal.",
        "Engineering note: rollout checklist and deployment runbook.",
        "Research summary: baseline model regressions and fix candidates.",
        "Support note: no critical incidents this week.",
    ]
    query = "What is the secret code for project atlas?"

    trace = await runtime.run_episode(
        goal=query,
        program="root_map_reduce",
        input_payload={
            "query": query,
            "documents": docs,
            "chunk_size": 2,
        },
    )

    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "demo"
    export_trace(trace, out_dir)
    export_cold_start([trace], out_dir / "cold_start_turns.jsonl")
    export_agentic_rl([trace], out_dir / "agentic_rl.jsonl")
    export_verl([trace], out_dir / "verl_warm_start.jsonl")
    export_openrlhf([trace], out_dir / "openrlhf_episodes.jsonl")

    print("Episode:", trace.episode_id)
    print("Success:", trace.success)
    print("Root output:")
    print(json.dumps(trace.root_output, ensure_ascii=False, indent=2))
    print("Artifacts:", out_dir)


if __name__ == "__main__":
    asyncio.run(main())
