from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, TYPE_CHECKING

from .types import MemoryObjectType, WriteReason, GroupStatus

if TYPE_CHECKING:
    from .runtime import GroupContext


ProgramFn = Callable[["GroupContext"], Awaitable[Any]]


class ProgramRegistry:
    def __init__(self) -> None:
        self._programs: Dict[str, ProgramFn] = {}

    def register(self, name: str, fn: ProgramFn) -> None:
        if name in self._programs:
            raise ValueError(f"Program already registered: {name}")
        self._programs[name] = fn

    def get(self, name: str) -> ProgramFn:
        if name not in self._programs:
            raise KeyError(f"Program not found: {name}")
        return self._programs[name]

    def list_programs(self) -> List[str]:
        return sorted(self._programs.keys())


async def extract_chunk_program(ctx: "GroupContext") -> Dict[str, Any]:
    query = str(ctx.input_payload.get("query", ""))
    documents = [str(x) for x in ctx.input_payload.get("documents", [])]
    chunk_index = int(ctx.input_payload.get("chunk_index", -1))

    extracted = await ctx.call_object(
        "extract_facts",
        {
            "query": query,
            "documents": documents,
            "max_hits": 8,
        },
    )

    facts = extracted.get("facts", [])
    confidence = min(1.0, 0.2 + 0.15 * len(facts))

    await ctx.write_memory(
        key=f"fact/{ctx.group_id}",
        object_type=MemoryObjectType.FACT,
        reason=WriteReason.VALUE_EVENT,
        content={
            "chunk_index": chunk_index,
            "facts": facts,
            "total_hits": extracted.get("total_hits", 0),
        },
        confidence=confidence,
    )

    return await ctx.finalize(
        {
            "chunk_index": chunk_index,
            "facts": facts,
            "task_score": 1.0 if facts else 0.1,
        }
    )


async def root_map_reduce_program(ctx: "GroupContext") -> Dict[str, Any]:
    query = str(ctx.input_payload.get("query", ""))
    documents = [str(x) for x in ctx.input_payload.get("documents", [])]
    chunk_size = max(1, int(ctx.input_payload.get("chunk_size", 3)))

    if not documents:
        return await ctx.finalize(
            {
                "answer": "No documents provided.",
                "facts": [],
                "task_score": 0.0,
            }
        )

    await ctx.write_memory(
        key=f"goal/{ctx.group_id}",
        object_type=MemoryObjectType.GOAL,
        reason=WriteReason.VALUE_EVENT,
        content={"query": query, "num_documents": len(documents), "chunk_size": chunk_size},
        confidence=1.0,
    )

    child_specs: List[Dict[str, Any]] = []
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i : i + chunk_size]
        child_specs.append(
            {
                "goal": f"Extract evidence from chunk {i // chunk_size}",
                "program": "extract_chunk",
                "input_payload": {
                    "query": query,
                    "documents": chunk,
                    "chunk_index": i // chunk_size,
                },
            }
        )

    child_ids = await ctx.spawn_groups(child_specs)
    child_results = await ctx.join_groups(child_ids)

    all_facts: List[Dict[str, Any]] = []
    failed_children = 0
    for child in child_results:
        if child.status == GroupStatus.SUCCEEDED and isinstance(child.output, dict):
            all_facts.extend(child.output.get("facts", []))
        else:
            failed_children += 1

    merged = await ctx.call_object(
        "merge_facts",
        {
            "query": query,
            "facts": all_facts,
        },
    )

    answer = str(merged.get("answer", ""))
    confidence = float(merged.get("confidence", 0.0))

    await ctx.write_memory(
        key=f"artifact/{ctx.group_id}",
        object_type=MemoryObjectType.ARTIFACT,
        reason=WriteReason.VALUE_EVENT,
        content={
            "answer": answer,
            "confidence": confidence,
            "facts_used": len(all_facts),
            "failed_children": failed_children,
        },
        confidence=confidence,
    )

    return await ctx.finalize(
        {
            "answer": answer,
            "confidence": confidence,
            "facts": all_facts,
            "failed_children": failed_children,
            "task_score": confidence,
        }
    )


def register_builtin_programs(registry: ProgramRegistry) -> None:
    registry.register("extract_chunk", extract_chunk_program)
    registry.register("root_map_reduce", root_map_reduce_program)
