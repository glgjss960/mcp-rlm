from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from .policy import build_policy_from_env
from .types import GroupStatus, MemoryObjectType, WriteReason

if TYPE_CHECKING:
    from .programs import ProgramRegistry
    from .runtime import GroupContext


async def mvp_leaf_segment_program(ctx: "GroupContext") -> Dict[str, Any]:
    query = str(ctx.input_payload.get("query", "")).strip()
    manifest_path = str(ctx.input_payload.get("manifest_path", "")).strip()
    segment_id = str(ctx.input_payload.get("segment_id", "")).strip()
    read_max_chars = int(ctx.input_payload.get("read_max_chars", 12000))

    if not query or not manifest_path or not segment_id:
        raise ValueError("mvp_leaf_segment requires query, manifest_path, and segment_id")

    policy = build_policy_from_env()
    segment_resp = await ctx.call_object(
        "ctx/read_segment",
        {
            "manifest_path": manifest_path,
            "segment_id": segment_id,
            "max_chars": read_max_chars,
        },
    )

    segment_meta = segment_resp.get("segment", {}) if isinstance(segment_resp, dict) else {}
    segment_text = str(segment_resp.get("text", "")) if isinstance(segment_resp, dict) else ""
    instruction = await policy.leaf_instruction(query=query, segment_meta=segment_meta)

    analysis = await ctx.call_object(
        "analysis/analyze_segment",
        {
            "query": query,
            "text": segment_text,
            "segment_id": segment_id,
            "instruction": instruction,
            "max_facts": 6,
        },
    )

    facts = analysis.get("facts", []) if isinstance(analysis, dict) else []
    confidence = float(analysis.get("confidence", 0.0)) if isinstance(analysis, dict) else 0.0

    await ctx.write_memory(
        key=f"fact/{ctx.group_id}",
        object_type=MemoryObjectType.FACT,
        reason=WriteReason.VALUE_EVENT,
        content={
            "segment_id": segment_id,
            "instruction": instruction,
            "facts": facts,
            "confidence": confidence,
        },
        confidence=confidence,
    )

    return await ctx.finalize(
        {
            "segment_id": segment_id,
            "facts": facts,
            "confidence": confidence,
            "task_score": confidence,
        }
    )


async def mvp_root_program(ctx: "GroupContext") -> Dict[str, Any]:
    query = str(ctx.input_payload.get("query", "")).strip()
    manifest_path = str(ctx.input_payload.get("manifest_path", "")).strip()
    max_children_override = ctx.input_payload.get("max_children")

    if not query or not manifest_path:
        raise ValueError("mvp_root requires query and manifest_path")

    policy = build_policy_from_env()
    stats = await ctx.call_object(
        "ctx/context_stats",
        {
            "manifest_path": manifest_path,
        },
    )
    plan = await policy.plan_root(query=query, context_stats=stats)

    max_children = plan.max_children
    if max_children_override is not None:
        max_children = max(1, min(int(max_children_override), plan.max_children))

    search = await ctx.call_object(
        "ctx/search_hierarchical",
        {
            "manifest_path": manifest_path,
            "query": query,
            "top_k": plan.top_k,
            "coarse_k": plan.coarse_k,
        },
    )
    hits = list(search.get("hits", [])) if isinstance(search, dict) else []

    await ctx.write_memory(
        key=f"goal/{ctx.group_id}",
        object_type=MemoryObjectType.GOAL,
        reason=WriteReason.VALUE_EVENT,
        content={
            "query": query,
            "manifest_path": manifest_path,
            "search_hits": len(hits),
            "policy_plan": {
                "top_k": plan.top_k,
                "coarse_k": plan.coarse_k,
                "max_children": max_children,
                "read_max_chars": plan.read_max_chars,
            },
        },
        confidence=1.0,
    )

    if not hits:
        return await ctx.finalize(
            {
                "answer": "No relevant segments found.",
                "confidence": 0.0,
                "facts": [],
                "task_score": 0.0,
            }
        )

    selected = hits[:max_children]
    child_specs: List[Dict[str, Any]] = []
    for hit in selected:
        segment_id = str(hit.get("segment_id", ""))
        if not segment_id:
            continue
        child_specs.append(
            {
                "goal": f"Analyze segment {segment_id}",
                "program": "mvp_leaf_segment",
                "input_payload": {
                    "query": query,
                    "manifest_path": manifest_path,
                    "segment_id": segment_id,
                    "read_max_chars": plan.read_max_chars,
                },
            }
        )

    child_ids = await ctx.spawn_groups(child_specs)
    child_results = await ctx.join_groups(child_ids)

    all_facts: List[Dict[str, Any]] = []
    failed_children = 0
    for child in child_results:
        if child.status == GroupStatus.SUCCEEDED and isinstance(child.output, dict):
            all_facts.extend(list(child.output.get("facts", [])))
        else:
            failed_children += 1

    merged = await ctx.call_object(
        "analysis/merge_facts",
        {
            "query": query,
            "facts": all_facts,
        },
    )

    final_policy = await policy.finalize_answer(query=query, merged=merged, facts=all_facts)
    answer = str(final_policy.get("answer", merged.get("answer", "")))
    confidence = float(final_policy.get("confidence", merged.get("confidence", 0.0)))

    output = {
        "answer": answer,
        "confidence": confidence,
        "facts": all_facts,
        "failed_children": failed_children,
        "policy": final_policy.get("policy", "unknown"),
        "task_score": confidence,
    }

    await ctx.write_memory(
        key=f"artifact/{ctx.group_id}",
        object_type=MemoryObjectType.ARTIFACT,
        reason=WriteReason.VALUE_EVENT,
        content=output,
        confidence=confidence,
    )

    return await ctx.finalize(output)


def register_mvp_programs(registry: "ProgramRegistry") -> None:
    registry.register("mvp_root", mvp_root_program)
    registry.register("mvp_leaf_segment", mvp_leaf_segment_program)
