from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING
import asyncio
import re

from .mcp import MCPCall
from .policy import build_policy_from_config
from .types import GroupStatus, MemoryObjectType, WriteReason

if TYPE_CHECKING:
    from .programs import ProgramRegistry
    from .runtime import GroupContext


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

_MCQ_LETTERS = ("A", "B", "C", "D")


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


def _keyword_query(query: str, *, max_terms: int = 14) -> str:
    terms = _tokenize_terms(query)
    if not terms:
        return query
    return " ".join(terms[:max_terms])


def _choice_map(payload: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    raw = payload.get("choices")
    if isinstance(raw, dict):
        for k, v in raw.items():
            letter = str(k).strip().upper()
            if letter in _MCQ_LETTERS:
                text = str(v).strip()
                if text:
                    out[letter] = text

    for letter in _MCQ_LETTERS:
        if letter in out:
            continue
        field = f"choice_{letter}"
        if field in payload:
            text = str(payload.get(field, "")).strip()
            if text:
                out[letter] = text
    return out


def _score_map(raw: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {k: 0.0 for k in _MCQ_LETTERS}
    data = raw.get("choice_scores", {})
    if not isinstance(data, dict):
        return out
    for letter in _MCQ_LETTERS:
        try:
            out[letter] = float(data.get(letter, 0.0))
        except (TypeError, ValueError):
            out[letter] = 0.0
    return out


def _merge_hits(weighted_sources: List[tuple[str, float, List[Dict[str, Any]]]], *, query: str, level0_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

        entry["score"] = float(entry["score"]) + (weight * base)
        sources = entry.get("sources", [])
        if source not in sources:
            sources.append(source)
        entry["sources"] = sources

    for source_name, weight, hits in weighted_sources:
        for hit in hits:
            if isinstance(hit, dict):
                add_hit(hit, weight=weight, source=source_name)

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
            weight=0.3,
            source="level0_preview",
        )

    merged = list(by_seg.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged


def _build_extra_calls(specs: Any) -> List[MCPCall]:
    calls: List[MCPCall] = []
    if not isinstance(specs, list):
        return calls
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        object_name = str(spec.get("object_name", "")).strip()
        if not object_name:
            continue
        payload = spec.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        timeout_seconds = float(spec.get("timeout_seconds", 20.0))
        calls.append(MCPCall(object_name=object_name, payload=payload, timeout_seconds=timeout_seconds))
    return calls


async def _best_effort_call_objects(ctx: "GroupContext", calls: List[MCPCall]) -> List[Dict[str, Any]]:
    async def one(call: MCPCall) -> Dict[str, Any]:
        try:
            output = await ctx.call_object(call.object_name, call.payload, timeout_seconds=call.timeout_seconds)
            return {
                "object_name": call.object_name,
                "ok": True,
                "output": output,
            }
        except Exception as exc:
            return {
                "object_name": call.object_name,
                "ok": False,
                "error": str(exc),
            }

    tasks = [asyncio.create_task(one(call)) for call in calls]
    return await asyncio.gather(*tasks)


async def longbench_v2_leaf_segment_program(ctx: "GroupContext") -> Dict[str, Any]:
    question = str(ctx.input_payload.get("question", "")).strip()
    manifest_path = str(ctx.input_payload.get("manifest_path", "")).strip()
    segment_id = str(ctx.input_payload.get("segment_id", "")).strip()
    read_max_chars = int(ctx.input_payload.get("read_max_chars", 12000))
    choices = _choice_map(ctx.input_payload)
    policy_config = ctx.input_payload.get("policy_config")

    extra_calls = _build_extra_calls(ctx.input_payload.get("leaf_extra_object_fanout", []))

    if not question or not manifest_path or not segment_id:
        raise ValueError("longbench_v2_leaf_segment requires question, manifest_path, and segment_id")

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
        "num_choices": len(choices),
    }

    windows_resp = await ctx.call_object(
        "ctx/read_segment_windows",
        {
            "manifest_path": manifest_path,
            "segment_id": segment_id,
            "window_chars": min(3600, max(800, read_max_chars // 3)),
            "stride_chars": min(2000, max(400, read_max_chars // 6)),
            "max_windows": 10,
            "query": question,
            "choices": choices,
        },
    )
    windows = windows_resp.get("windows", []) if isinstance(windows_resp, dict) else []

    instruction = await policy.leaf_instruction(query=question, segment_meta=segment_meta)

    leaf_calls = [
        MCPCall(
            object_name="analysis/analyze_segment",
            payload={
                "query": question,
                "text": segment_text,
                "segment_id": segment_id,
                "instruction": instruction,
                "max_facts": 8,
            },
        ),
        MCPCall(
            object_name="analysis/score_mcq_choices",
            payload={
                "question": question,
                "choices": choices,
                "text": segment_text,
                "segment_id": segment_id,
                "max_evidence": 4,
            },
        ),
        MCPCall(
            object_name="analysis/score_mcq_windows",
            payload={
                "question": question,
                "choices": choices,
                "windows": windows,
                "segment_id": segment_id,
                "max_windows": 8,
                "max_evidence": 24,
            },
        ),
        MCPCall(
            object_name="analysis/eliminate_choices",
            payload={
                "question": question,
                "choices": choices,
                "text": segment_text,
                "eliminate_threshold": 2.2,
            },
        ),
        MCPCall(
            object_name="analysis/extract_code_cues",
            payload={
                "question": question,
                "choices": choices,
                "text": segment_text,
                "max_evidence": 18,
            },
        ),
        MCPCall(
            object_name="analysis/extract_table_cues",
            payload={
                "question": question,
                "choices": choices,
                "text": segment_text,
                "max_evidence": 18,
            },
        ),
    ]

    primary_results = await ctx.call_objects(leaf_calls)

    analyze = primary_results[0] if len(primary_results) > 0 and isinstance(primary_results[0], dict) else {}
    full_score = primary_results[1] if len(primary_results) > 1 and isinstance(primary_results[1], dict) else {}
    window_score = primary_results[2] if len(primary_results) > 2 and isinstance(primary_results[2], dict) else {}
    elimination = primary_results[3] if len(primary_results) > 3 and isinstance(primary_results[3], dict) else {}
    code_cues = primary_results[4] if len(primary_results) > 4 and isinstance(primary_results[4], dict) else {}
    table_cues = primary_results[5] if len(primary_results) > 5 and isinstance(primary_results[5], dict) else {}

    extra_results = await _best_effort_call_objects(ctx, extra_calls) if extra_calls else []
    extra_score_maps: List[Dict[str, Any]] = []
    extra_evidence: List[Dict[str, Any]] = []
    for item in extra_results:
        if not isinstance(item, dict) or not bool(item.get("ok")):
            continue
        output = item.get("output", {})
        if isinstance(output, dict) and isinstance(output.get("choice_scores"), dict):
            extra_score_maps.append(output)
        if isinstance(output, dict) and isinstance(output.get("evidence"), list):
            for ev in output.get("evidence", []):
                if isinstance(ev, dict):
                    extra_evidence.append(ev)

    vote = await ctx.call_object(
        "analysis/vote_choice_scores",
        {
            "maps": [full_score, window_score, code_cues, table_cues] + extra_score_maps,
            "weights": [1.25, 1.15, 0.65, 0.65] + [0.5 for _ in extra_score_maps],
            "elimination_penalties": elimination.get("choice_penalties", {}),
            "elimination_weight": 1.25,
        },
    )

    choice_scores = _score_map(vote if isinstance(vote, dict) else {})
    confidence = float(vote.get("confidence", 0.0)) if isinstance(vote, dict) else 0.0

    evidence: List[Dict[str, Any]] = []
    for source in [full_score, window_score, code_cues, table_cues]:
        if isinstance(source, dict) and isinstance(source.get("evidence", []), list):
            for ev in source.get("evidence", []):
                if isinstance(ev, dict):
                    evidence.append(ev)
    evidence.extend(extra_evidence)
    evidence.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    output = {
        "segment_id": segment_id,
        "choice_scores": {k: round(float(v), 4) for k, v in choice_scores.items()},
        "choice_penalties": elimination.get("choice_penalties", {}) if isinstance(elimination, dict) else {},
        "eliminated": elimination.get("eliminated", []) if isinstance(elimination, dict) else [],
        "evidence": evidence[:40],
        "facts": list(analyze.get("facts", [])) if isinstance(analyze.get("facts", []), list) else [],
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "task_score": round(max(0.0, min(1.0, confidence)), 3),
    }
    ctx.local_state["leaf_vote_summary"] = {
        "best_choice": max(choice_scores, key=choice_scores.get) if choice_scores else "",
        "confidence": output["confidence"],
        "evidence_items": len(output["evidence"]),
    }

    if ctx.context_usage() >= 0.72:
        await ctx.flush_context_pressure(note="longbench_leaf_post_scoring")

    await ctx.write_memory(
        key=f"fact/{ctx.group_id}",
        object_type=MemoryObjectType.FACT,
        reason=WriteReason.VALUE_EVENT,
        content={
            "segment_id": segment_id,
            "instruction": instruction,
            "choice_scores": output["choice_scores"],
            "choice_penalties": output["choice_penalties"],
            "eliminated": output["eliminated"],
            "parallel_objects": [
                "analysis/analyze_segment",
                "analysis/score_mcq_choices",
                "analysis/score_mcq_windows",
                "analysis/eliminate_choices",
                "analysis/extract_code_cues",
                "analysis/extract_table_cues",
                "analysis/vote_choice_scores",
            ] + [c.object_name for c in extra_calls],
        },
        confidence=output["confidence"],
    )

    return await ctx.finalize(output)


async def longbench_v2_root_program(ctx: "GroupContext") -> Dict[str, Any]:
    question = str(ctx.input_payload.get("question", "")).strip()
    manifest_path = str(ctx.input_payload.get("manifest_path", "")).strip()
    max_children_override = ctx.input_payload.get("max_children")
    policy_config = ctx.input_payload.get("policy_config")
    choices = _choice_map(ctx.input_payload)

    root_extra_calls = _build_extra_calls(ctx.input_payload.get("root_extra_object_fanout", []))

    if not question or not manifest_path:
        raise ValueError("longbench_v2_root requires question and manifest_path")
    if len(choices) < 2:
        raise ValueError("longbench_v2_root requires at least 2 choices")

    resolved_policy_config = policy_config if isinstance(policy_config, dict) else None
    policy = build_policy_from_config(resolved_policy_config)

    stats = await ctx.call_object(
        "ctx/context_stats",
        {
            "manifest_path": manifest_path,
        },
    )
    plan = await policy.plan_root(query=question, context_stats=stats)

    max_children = plan.max_children
    if max_children_override is not None:
        max_children = max(1, min(int(max_children_override), max(4, plan.max_children * 2)))

    keyword_query = _keyword_query(question)
    choice_queries = [f"{question} {text}" for _, text in sorted(choices.items())]

    retrieval_calls: List[MCPCall] = [
        MCPCall(
            object_name="ctx/search_hierarchical",
            payload={
                "manifest_path": manifest_path,
                "query": question,
                "top_k": max(plan.top_k, max_children),
                "coarse_k": plan.coarse_k,
            },
        ),
        MCPCall(
            object_name="ctx/search_hierarchical",
            payload={
                "manifest_path": manifest_path,
                "query": keyword_query,
                "top_k": min(64, max(plan.top_k + 8, max_children)),
                "coarse_k": max(8, plan.coarse_k // 2),
            },
        ),
        MCPCall(
            object_name="ctx/search_hierarchical_mmr",
            payload={
                "manifest_path": manifest_path,
                "query": question,
                "top_k": min(64, max(plan.top_k + 8, max_children)),
                "coarse_k": min(64, max(plan.coarse_k, plan.coarse_k * 2)),
                "candidate_k": min(128, max(24, plan.top_k * 5)),
                "lambda_mult": 0.72,
            },
        ),
        MCPCall(
            object_name="ctx/search_multi_query",
            payload={
                "manifest_path": manifest_path,
                "queries": [question, keyword_query] + choice_queries,
                "top_k": min(96, max(plan.top_k * 2, max_children * 2)),
                "per_query_top_k": min(24, max(8, plan.top_k // 2)),
                "coarse_k": min(64, max(plan.coarse_k, plan.coarse_k * 2)),
            },
        ),
        MCPCall(
            object_name="ctx/list_level",
            payload={
                "manifest_path": manifest_path,
                "level": 0,
                "limit": max(24, max_children * 6),
                "offset": 0,
            },
        ),
    ]

    for choice_letter, choice_text in sorted(choices.items()):
        retrieval_calls.append(
            MCPCall(
                object_name="ctx/search_hierarchical",
                payload={
                    "manifest_path": manifest_path,
                    "query": f"{question} {choice_text}",
                    "top_k": min(40, max(10, plan.top_k // 2)),
                    "coarse_k": max(8, plan.coarse_k // 2),
                    "tag": choice_letter,
                },
            )
        )

    retrieval_results = await ctx.call_objects(retrieval_calls)

    weighted_sources: List[tuple[str, float, List[Dict[str, Any]]]] = []
    level0_segments: List[Dict[str, Any]] = []

    for idx, result in enumerate(retrieval_results):
        if not isinstance(result, dict):
            continue
        if idx == 0:
            weighted_sources.append(("primary", 1.0, list(result.get("hits", []))))
        elif idx == 1:
            weighted_sources.append(("keyword", 0.86, list(result.get("hits", []))))
        elif idx == 2:
            weighted_sources.append(("mmr", 0.84, list(result.get("hits", []))))
        elif idx == 3:
            weighted_sources.append(("multi_query", 0.9, list(result.get("hits", []))))
        elif idx == 4:
            level0_segments = list(result.get("segments", []))
        else:
            weighted_sources.append((f"choice_{idx-4}", 0.66, list(result.get("hits", []))))

    merged_hits = _merge_hits(weighted_sources, query=question, level0_segments=level0_segments)

    reranked = await ctx.call_object(
        "analysis/rerank_hits_with_choices",
        {
            "query": question,
            "choices": choices,
            "hits": merged_hits,
            "top_k": max(max_children * 4, 24),
        },
    )
    ranked_hits = list(reranked.get("hits", [])) if isinstance(reranked, dict) else merged_hits

    ctx.local_state["root_retrieval_summary"] = {
        "question": question,
        "merged_hits": len(merged_hits),
        "ranked_hits": len(ranked_hits),
        "top_segments": [str(h.get("segment_id", "")) for h in ranked_hits[:8] if isinstance(h, dict)],
    }

    if ctx.context_usage() >= 0.72:
        await ctx.flush_context_pressure(note="longbench_root_post_retrieval")

    extra_results = await _best_effort_call_objects(ctx, root_extra_calls) if root_extra_calls else []

    await ctx.write_memory(
        key=f"goal/{ctx.group_id}",
        object_type=MemoryObjectType.GOAL,
        reason=WriteReason.VALUE_EVENT,
        content={
            "question": question,
            "choices": choices,
            "manifest_path": manifest_path,
            "merged_hits": len(merged_hits),
            "ranked_hits": len(ranked_hits),
            "root_parallel_calls": [c.object_name for c in retrieval_calls] + [c.object_name for c in root_extra_calls],
            "root_extra_results": extra_results,
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

    if not ranked_hits:
        fallback = "A" if "A" in choices else sorted(choices.keys())[0]
        return await ctx.finalize(
            {
                "pred": fallback,
                "response": f"The correct answer is ({fallback})",
                "confidence": 0.0,
                "choice_scores": {k: 0.0 for k in _MCQ_LETTERS},
                "evidence": [],
                "failed_children": 0,
                "task_score": 0.0,
            }
        )

    selected = ranked_hits[:max_children]
    child_specs: List[Dict[str, Any]] = []
    for hit in selected:
        segment_id = str(hit.get("segment_id", "")).strip()
        if not segment_id:
            continue
        child_specs.append(
            {
                "goal": f"Score choices on segment {segment_id}",
                "program": "longbench_v2_leaf_segment",
                "input_payload": {
                    "question": question,
                    "choices": choices,
                    "manifest_path": manifest_path,
                    "segment_id": segment_id,
                    "read_max_chars": plan.read_max_chars,
                    "policy_config": resolved_policy_config,
                    "leaf_extra_object_fanout": ctx.input_payload.get("leaf_extra_object_fanout", []),
                },
            }
        )

    if ctx.context_usage() >= 0.72:
        await ctx.flush_context_pressure(note="longbench_root_before_spawn")

    child_ids = await ctx.spawn_groups(child_specs)
    child_results = await ctx.join_groups(child_ids)

    leaf_outputs: List[Dict[str, Any]] = []
    failed_children = 0
    global_penalties = {k: 0.0 for k in _MCQ_LETTERS}
    for child in child_results:
        if child.status == GroupStatus.SUCCEEDED and isinstance(child.output, dict):
            leaf_outputs.append(child.output)
            penalties = child.output.get("choice_penalties", {})
            if isinstance(penalties, dict):
                for k in _MCQ_LETTERS:
                    try:
                        global_penalties[k] += float(penalties.get(k, 0.0))
                    except (TypeError, ValueError):
                        continue
        else:
            failed_children += 1

    ctx.local_state["join_summary"] = {
        "children": len(child_results),
        "successful_children": len(leaf_outputs),
        "failed_children": failed_children,
    }

    if ctx.context_usage() >= 0.72:
        await ctx.flush_context_pressure(note="longbench_root_post_join")

    aggregate = await ctx.call_object(
        "analysis/aggregate_mcq_scores",
        {
            "items": leaf_outputs,
            "max_evidence": 48,
        },
    )

    voted = await ctx.call_object(
        "analysis/vote_choice_scores",
        {
            "maps": [aggregate],
            "weights": [1.0],
            "elimination_penalties": global_penalties,
            "elimination_weight": 0.25,
        },
    )

    choice_scores = _score_map(voted if isinstance(voted, dict) else {})
    fallback_choice = str(voted.get("best_choice", "")).strip().upper() if isinstance(voted, dict) else ""
    if fallback_choice not in _MCQ_LETTERS:
        ranked = sorted(choice_scores.items(), key=lambda kv: kv[1], reverse=True)
        fallback_choice = ranked[0][0] if ranked else ("A" if "A" in choices else sorted(choices.keys())[0])

    aggregate_conf = float(voted.get("confidence", 0.0)) if isinstance(voted, dict) else 0.0
    evidence = list(aggregate.get("evidence", [])) if isinstance(aggregate, dict) and isinstance(aggregate.get("evidence", []), list) else []

    merged = {
        "answer": f"Candidate best option: {fallback_choice}",
        "confidence": aggregate_conf,
        "choice_scores": choice_scores,
    }
    facts_for_policy = [
        {
            "text": str(ev.get("text", "")),
            "score": float(ev.get("score", 0.0)),
            "choice": str(ev.get("choice", "")),
        }
        for ev in evidence[:32]
        if isinstance(ev, dict)
    ]

    choice_lines = "\n".join([f"({k}) {v}" for k, v in sorted(choices.items())])
    query_for_policy = (
        "You are solving a multiple-choice question.\n"
        "Return ONLY one final answer in this exact format: The correct answer is (X).\n\n"
        f"Question: {question}\n"
        f"Choices:\n{choice_lines}\n"
        f"Fallback candidate: {fallback_choice}"
    )

    final_policy = await policy.finalize_answer(query=query_for_policy, merged=merged, facts=facts_for_policy)
    raw_answer = str(final_policy.get("answer", ""))

    normalized = await ctx.call_object(
        "analysis/normalize_mcq_answer",
        {
            "response": raw_answer,
            "fallback": fallback_choice,
        },
    )

    pred = str(normalized.get("answer", fallback_choice)).strip().upper()
    if pred not in _MCQ_LETTERS:
        pred = fallback_choice

    response = str(normalized.get("normalized_response", "")).strip()
    if not response:
        response = f"The correct answer is ({pred})"

    final_conf = max(aggregate_conf, float(final_policy.get("confidence", 0.0)))
    output = {
        "pred": pred,
        "response": response,
        "confidence": round(max(0.0, min(1.0, final_conf)), 3),
        "choice_scores": {k: round(float(v), 4) for k, v in choice_scores.items()},
        "choice_penalties": {k: round(float(v), 4) for k, v in global_penalties.items()},
        "evidence": evidence,
        "failed_children": failed_children,
        "task_score": round(max(0.0, min(1.0, final_conf)), 3),
    }

    await ctx.write_memory(
        key=f"artifact/{ctx.group_id}",
        object_type=MemoryObjectType.ARTIFACT,
        reason=WriteReason.VALUE_EVENT,
        content=output,
        confidence=output["confidence"],
    )

    return await ctx.finalize(output)


def register_longbench_v2_programs(registry: "ProgramRegistry") -> None:
    registry.register("longbench_v2_root", longbench_v2_root_program)
    registry.register("longbench_v2_leaf_segment", longbench_v2_leaf_segment_program)
