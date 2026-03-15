from __future__ import annotations

from pathlib import Path
import argparse
import asyncio
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_rlm.training import (
    export_agentic_rl,
    export_cold_start,
    export_openrlhf,
    export_traces,
    export_verl,
    synthesize_traces,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize MCP-RLM trajectories")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="artifacts/synth")
    args = parser.parse_args()

    traces = await synthesize_traces(
        num_episodes=args.episodes,
        parallelism=args.parallelism,
        seed=args.seed,
    )

    out_dir = Path(__file__).resolve().parents[1] / args.out
    export_traces(traces, out_dir)
    export_cold_start(traces, out_dir / "cold_start_turns.jsonl")
    export_agentic_rl(traces, out_dir / "agentic_rl.jsonl")
    export_verl(traces, out_dir / "verl_warm_start.jsonl")
    export_openrlhf(traces, out_dir / "openrlhf_episodes.jsonl")

    solved = 0
    for trace in traces:
        output = trace.root_output if isinstance(trace.root_output, dict) else {}
        if output.get("is_correct"):
            solved += 1

    print(f"Generated episodes: {len(traces)}")
    print(f"Correct episodes: {solved}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
