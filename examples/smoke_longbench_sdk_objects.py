from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import argparse
import asyncio
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_rlm import MCPCall, MCPInvocationContext, MCPServerSpec, MultiServerMCPClient, preprocess_long_context


async def _call(client: MultiServerMCPClient, object_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    result = await client.call(
        MCPCall(object_name=object_name, payload=payload),
        MCPInvocationContext(episode_id="smoke", group_id="root"),
    )
    if not result.ok:
        raise RuntimeError(f"{object_name} failed: {result.error}")
    if not isinstance(result.output, dict):
        raise RuntimeError(f"{object_name} returned non-dict output")
    return result.output


async def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SDK-mode smoke for key LongBench MCP objects")
    parser.add_argument("--context-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--choice-a", type=str, required=True)
    parser.add_argument("--choice-b", type=str, required=True)
    parser.add_argument("--choice-c", type=str, required=True)
    parser.add_argument("--choice-d", type=str, required=True)
    parser.add_argument("--out", type=str, default="artifacts/smoke_longbench_sdk")
    parser.add_argument("--legacy-mcp", action="store_true")
    parser.add_argument("--require-official-mcp-sdk", action="store_true")
    args = parser.parse_args()

    context_file = Path(args.context_file).resolve()
    if not context_file.exists():
        raise FileNotFoundError(f"Context file not found: {context_file}")

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    store_dir = out_dir / "context_store"
    manifest = preprocess_long_context(
        input_file=context_file,
        output_dir=store_dir,
        chunk_chars=16000,
        overlap_chars=400,
        branch_factor=8,
    )

    ctx_server = ROOT / "examples" / "run_context_server.py"
    analysis_server = ROOT / "examples" / "run_analysis_server.py"

    ctx_cmd = [sys.executable, str(ctx_server), "--manifest", str(manifest)]
    analysis_cmd = [sys.executable, str(analysis_server)]
    if args.legacy_mcp:
        ctx_cmd.append("--legacy-mcp")
        analysis_cmd.append("--legacy-mcp")
    if args.require_official_mcp_sdk:
        ctx_cmd.append("--require-official-sdk")
        analysis_cmd.append("--require-official-sdk")

    client = MultiServerMCPClient(
        specs=[
            MCPServerSpec(
                alias="ctx",
                command=ctx_cmd,
                cwd=str(ROOT),
                prefer_official_sdk=not args.legacy_mcp,
                strict_official_sdk=bool(args.require_official_mcp_sdk),
            ),
            MCPServerSpec(
                alias="analysis",
                command=analysis_cmd,
                cwd=str(ROOT),
                prefer_official_sdk=not args.legacy_mcp,
                strict_official_sdk=bool(args.require_official_mcp_sdk),
            ),
        ]
    )

    choices = {
        "A": args.choice_a,
        "B": args.choice_b,
        "C": args.choice_c,
        "D": args.choice_d,
    }

    try:
        stats = await _call(client, "ctx/context_stats", {"manifest_path": str(manifest)})
        search = await _call(
            client,
            "ctx/search_hierarchical",
            {
                "manifest_path": str(manifest),
                "query": args.query,
                "top_k": 4,
                "coarse_k": 24,
            },
        )

        hits = [x for x in search.get("hits", []) if isinstance(x, dict)]
        if not hits:
            raise RuntimeError("ctx/search_hierarchical returned no hits")
        segment_id = str(hits[0].get("segment_id", "")).strip()
        if not segment_id:
            raise RuntimeError("top hit has no segment_id")

        segment = await _call(
            client,
            "ctx/read_segment_adaptive",
            {
                "manifest_path": str(manifest),
                "segment_id": segment_id,
                "question": args.query,
                "choices": choices,
                "max_chars": 6000,
                "max_windows": 6,
            },
        )

        score = await _call(
            client,
            "analysis/score_mcq_choices",
            {
                "question": args.query,
                "choices": choices,
                "text": str(segment.get("text", "")),
                "windows": segment.get("windows", []),
                "segment_id": segment_id,
            },
        )

        summary = {
            "manifest": str(manifest),
            "stats_ok": True,
            "search_ok": True,
            "score_ok": True,
            "top_segment_id": segment_id,
            "stats": {
                "total_chars": stats.get("total_chars"),
                "leaf_segments": stats.get("leaf_segments"),
                "semantic_enabled": stats.get("semantic_enabled"),
            },
            "search_top": hits[:2],
            "score": {
                "best_choice": score.get("best_choice"),
                "confidence": score.get("confidence"),
                "choice_scores": score.get("choice_scores"),
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
