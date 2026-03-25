from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING
import re

from .mcp import MCPCall
from .policy import build_policy_from_config
from .types import GroupStatus, MemoryObjectType, WriteReason

if TYPE_CHECKING:
    from .programs import ProgramRegistry
    from .runtime import GroupContext


def _normalize_fact(raw: Dict[str, Any], *, segment_id: str) -> Dict[str, Any] | None:
    text = str(raw.get("text", "")).strip()
    if not text:
        return None
    score = float(raw.get("score", 0.0))
    out: Dict[str, Any] = {
        "segment_id": segment_id,
        "text": text,
        "score": score,
    }
    if "doc_id" in raw:
        out["doc_id"] = raw["doc_id"]
    return out


def _dedupe_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_text: Dict[str, Dict[str, Any]] = {}
    for fact in facts:
        text = str(fact.get("text", "")).strip()
        if not text:
            continue
        existing = best_by_text.get(text)
        if existing is None or float(fact.get("score", 0.0)) > float(existing.get("score", 0.0)):
            best_by_text[text] = fact
    merged = list(best_by_text.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "why",
    "with",
}


def _tokenize_terms(text: str) -> List[str]:
    terms = [t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(t) > 1]
    dedup: List[str] = []
    seen = set()
    for term in terms:
        if term in _STOPWORDS:
            continue
        if term in seen:
            continue
        seen.add(term)
        dedup.append(term)
    return dedup


def _keyword_query(query: str, *, max_terms: int = 12) -> str:
    terms = _tokenize_terms(query)
    if not terms:
        return query
    return " ".join(terms[:max_terms])


def _merge_root_hits(
    *,
    query: str,
    primary_hits: List[Dict[str, Any]],
    keyword_hits: List[Dict[str, Any]],
    recall_hits: List[Dict[str, Any]],
    level0_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_seg: Dict[str, Dict[str, Any]] = {}

    def add_hit(raw: Dict[str, Any], *, weight: float, source: str) -> None:
        seg_id = str(raw.get("segment_id", "")).strip()
        if not seg_id:
            return
        base = float(raw.get("score", 0.0))
        if base <= 0:
            base = 0.1
        entry = by_seg.get(seg_id)
        if entry is None:
            entry = {
                "segment_id": seg_id,
                "score": 0.0,
                "preview": str(raw.get("preview", "")),
                "start_char": raw.get("start_char"),
                "end_char": raw.get("end_char"),
                "num_chars": raw.get("num_chars"),
                "sources": [],
            }
            by_seg[seg_id] = entry
        entry["score"] = float(entry["score"]) + weight * base
        sources = entry.get("sources", [])
        if source not in sources:
            sources.append(source)
        entry["sources"] = sources

    for hit in primary_hits:
        if isinstance(hit, dict):
            add_hit(hit, weight=1.0, source="primary")
    for hit in keyword_hits:
        if isinstance(hit, dict):
            add_hit(hit, weight=0.85, source="keyword")
    for hit in recall_hits:
        if isinstance(hit, dict):
            add_hit(hit, weight=0.7, source="recall")

    query_terms = _tokenize_terms(query)
    for seg in level0_segments:
        if not isinstance(seg, dict):
            continue
        seg_id = str(seg.get("segment_id", "")).strip()
        if not seg_id:
            continue
        preview = str(seg.get("preview", "")).lower()
        overlap = sum(1 for t in query_terms if t in preview)
        if overlap <= 0:
            continue
        add_hit(
            {
                "segment_id": seg_id,
                "score": float(overlap),
                "preview": seg.get("preview"),
                "start_char": seg.get("start_char"),
                "end_char": seg.get("end_char"),
                "num_chars": seg.get("num_chars"),
            },
            weight=0.35,
            source="level0_preview",
        )

    merged = list(by_seg.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged


async def mvp_leaf_segment_program(ctx: "GroupContext") -> Dict[str, Any]:
    query = str(ctx.input_payload.get("query", "")).strip()
    manifest_path = str(ctx.input_payload.get("manifest_path", "")).strip()
    segment_id = str(ctx.input_payload.get("segment_id", "")).strip()
    read_max_chars = int(ctx.input_payload.get("read_max_chars", 12000))
    policy_config = ctx.input_payload.get("policy_config")

    if not query or not manifest_path or not segment_id:
        raise ValueError("mvp_leaf_segment requires query, manifest_path, and segment_id")

    policy = build_policy_from_config(policy_config if isinstance(policy_config, dict) else None)
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
    ctx.local_state["segment_context"] = {
        "segment_id": segment_id,
        "segment_chars": len(segment_text),
    }
    instruction = await policy.leaf_instruction(query=query, segment_meta=segment_meta)

    results = await ctx.call_objects(
        [
            MCPCall(
                object_name="analysis/analyze_segment",
                payload={
                    "query": query,
                    "text": segment_text,
                    "segment_id": segment_id,
                    "instruction": instruction,
                    "max_facts": 6,
                },
            ),
            MCPCall(
                object_name="analysis/extract_facts",
                payload={
                    "query": query,
                    "documents": [segment_text],
                    "max_hits": 6,
                },
            ),
        ]
    )

    analyze = results[0] if len(results) > 0 and isinstance(results[0], dict) else {}
    extract = results[1] if len(results) > 1 and isinstance(results[1], dict) else {}

    combined: List[Dict[str, Any]] = []
    for raw in list(analyze.get("facts", [])):
        if isinstance(raw, dict):
            normalized = _normalize_fact(raw, segment_id=segment_id)
            if normalized is not None:
                combined.append(normalized)
    for raw in list(extract.get("facts", [])):
        if isinstance(raw, dict):
            normalized = _normalize_fact(raw, segment_id=segment_id)
            if normalized is not None:
                combined.append(normalized)

    facts = _dedupe_facts(combined)[:8]
    ctx.local_state["leaf_fact_summary"] = {
        "num_facts": len(facts),
        "top_fact": str(facts[0].get("text", ""))[:220] if facts else "",
    }

    analyze_conf = float(analyze.get("confidence", 0.0)) if isinstance(analyze, dict) else 0.0
    best_score = float(facts[0].get("score", 0.0)) if facts else 0.0
    denom = max(1.0, float(len(query.split())))
    extract_conf = min(1.0, best_score / denom)
    confidence = max(analyze_conf, extract_conf)

    if ctx.context_usage() >= 0.72:
        await ctx.flush_context_pressure(note="mvp_leaf_post_analysis")

    await ctx.write_memory(
        key=f"fact/{ctx.group_id}",
        object_type=MemoryObjectType.FACT,
        reason=WriteReason.VALUE_EVENT,
        content={
            "segment_id": segment_id,
            "instruction": instruction,
            "facts": facts,
            "confidence": confidence,
            "parallel_objects": ["analysis/analyze_segment", "analysis/extract_facts"],
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
    policy_config = ctx.input_payload.get("policy_config")

    if not query or not manifest_path:
        raise ValueError("mvp_root requires query and manifest_path")

    resolved_policy_config = policy_config if isinstance(policy_config, dict) else None
    policy = build_policy_from_config(resolved_policy_config)
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

    keyword_query = _keyword_query(query)
    root_calls = [
        MCPCall(
            object_name="ctx/search_hierarchical",
            payload={
                "manifest_path": manifest_path,
                "query": query,
                "top_k": plan.top_k,
                "coarse_k": plan.coarse_k,
            },
        ),
        MCPCall(
            object_name="ctx/search_hierarchical",
            payload={
                "manifest_path": manifest_path,
                "query": keyword_query,
                "top_k": min(48, max(plan.top_k, 8)),
                "coarse_k": max(8, plan.coarse_k // 2),
            },
        ),
        MCPCall(
            object_name="ctx/search_hierarchical",
            payload={
                "manifest_path": manifest_path,
                "query": query,
                "top_k": min(64, max(plan.top_k * 2, 12)),
                "coarse_k": min(64, max(plan.coarse_k, plan.coarse_k * 2)),
            },
        ),
        MCPCall(
            object_name="ctx/list_level",
            payload={
                "manifest_path": manifest_path,
                "level": 0,
                "limit": max(16, max_children * 4),
                "offset": 0,
            },
        ),
    ]
    root_results = await ctx.call_objects(root_calls)

    primary = root_results[0] if len(root_results) > 0 and isinstance(root_results[0], dict) else {}
    keyword = root_results[1] if len(root_results) > 1 and isinstance(root_results[1], dict) else {}
    recall = root_results[2] if len(root_results) > 2 and isinstance(root_results[2], dict) else {}
    level0 = root_results[3] if len(root_results) > 3 and isinstance(root_results[3], dict) else {}

    hits = _merge_root_hits(
        query=query,
        primary_hits=list(primary.get("hits", [])),
        keyword_hits=list(keyword.get("hits", [])),
        recall_hits=list(recall.get("hits", [])),
        level0_segments=list(level0.get("segments", [])),
    )
    ctx.local_state["root_retrieval_summary"] = {
        "query": query,
        "hits": len(hits),
        "top_segments": [str(h.get("segment_id", "")) for h in hits[:6]],
    }

    if ctx.context_usage() >= 0.72:
        await ctx.flush_context_pressure(note="mvp_root_post_retrieval")

    await ctx.write_memory(
        key=f"goal/{ctx.group_id}",
        object_type=MemoryObjectType.GOAL,
        reason=WriteReason.VALUE_EVENT,
        content={
            "query": query,
            "manifest_path": manifest_path,
            "search_hits": len(hits),
            "root_fanout": {
                "calls": [c.object_name for c in root_calls],
                "keyword_query": keyword_query,
                "primary_hits": len(primary.get("hits", [])) if isinstance(primary, dict) else 0,
                "keyword_hits": len(keyword.get("hits", [])) if isinstance(keyword, dict) else 0,
                "recall_hits": len(recall.get("hits", [])) if isinstance(recall, dict) else 0,
                "level0_segments": len(level0.get("segments", [])) if isinstance(level0, dict) else 0,
            },
            "policy_config": resolved_policy_config or {"mode": "heuristic"},
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
                    "policy_config": resolved_policy_config,
                },
            }
        )

    if ctx.context_usage() >= 0.72:
        await ctx.flush_context_pressure(note="mvp_root_before_spawn")

    child_ids = await ctx.spawn_groups(child_specs)
    child_results = await ctx.join_groups(child_ids)

    all_facts: List[Dict[str, Any]] = []
    failed_children = 0
    for child in child_results:
        if child.status == GroupStatus.SUCCEEDED and isinstance(child.output, dict):
            all_facts.extend(list(child.output.get("facts", [])))
        else:
            failed_children += 1

    ctx.local_state["join_summary"] = {
        "children": len(child_results),
        "failed_children": failed_children,
        "all_facts": len(all_facts),
    }

    if ctx.context_usage() >= 0.72:
        await ctx.flush_context_pressure(note="mvp_root_post_join")

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
