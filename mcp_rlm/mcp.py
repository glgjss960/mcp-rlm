from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Awaitable, Callable, Dict, List, Optional
import asyncio
import inspect
import re


@dataclass
class MCPInvocationContext:
    episode_id: str
    group_id: str


@dataclass
class MCPCall:
    object_name: str
    payload: Dict[str, Any]
    timeout_seconds: float = 30.0


@dataclass
class MCPResult:
    object_name: str
    ok: bool
    output: Any = None
    error: Optional[str] = None
    latency_ms: int = 0


MCPHandler = Callable[[Dict[str, Any], MCPInvocationContext], Any]


class MCPCallError(RuntimeError):
    pass


class MCPRegistry:
    def __init__(self) -> None:
        self._handlers: Dict[str, MCPHandler] = {}

    def register(self, name: str, handler: MCPHandler) -> None:
        if name in self._handlers:
            raise ValueError(f"MCP object already registered: {name}")
        self._handlers[name] = handler

    def get(self, name: str) -> MCPHandler:
        if name not in self._handlers:
            raise KeyError(f"MCP object not found: {name}")
        return self._handlers[name]

    def list_objects(self) -> List[str]:
        return sorted(self._handlers.keys())


class MCPClient:
    def __init__(self, registry: MCPRegistry, *, max_concurrency: int = 32) -> None:
        self._registry = registry
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def call(self, call: MCPCall, ctx: MCPInvocationContext) -> MCPResult:
        start = perf_counter()
        try:
            async with self._semaphore:
                handler = self._registry.get(call.object_name)
                raw_result = handler(call.payload, ctx)
                if inspect.isawaitable(raw_result):
                    output = await asyncio.wait_for(raw_result, timeout=call.timeout_seconds)
                else:
                    output = raw_result
            return MCPResult(
                object_name=call.object_name,
                ok=True,
                output=output,
                latency_ms=int((perf_counter() - start) * 1000),
            )
        except Exception as exc:
            return MCPResult(
                object_name=call.object_name,
                ok=False,
                error=str(exc),
                latency_ms=int((perf_counter() - start) * 1000),
            )

    async def call_many(self, calls: List[MCPCall], ctx: MCPInvocationContext) -> List[MCPResult]:
        tasks = [asyncio.create_task(self.call(call, ctx)) for call in calls]
        return await asyncio.gather(*tasks)


def _split_sentences(text: str) -> List[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]


def _extract_facts(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    query = str(payload.get("query", "")).strip().lower()
    docs = [str(x) for x in payload.get("documents", [])]
    max_hits = int(payload.get("max_hits", 5))
    query_terms = [t for t in re.findall(r"[a-zA-Z0-9_]+", query) if len(t) > 1]

    hits = []
    for doc_id, doc in enumerate(docs):
        for sentence in _split_sentences(doc):
            low = sentence.lower()
            score = sum(1 for term in query_terms if term in low)
            if score > 0:
                hits.append({"doc_id": doc_id, "text": sentence, "score": score})

    hits.sort(key=lambda x: x["score"], reverse=True)
    return {
        "facts": hits[:max_hits],
        "total_hits": len(hits),
        "query_terms": query_terms,
    }


def _merge_facts(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    query = str(payload.get("query", "")).strip()
    facts = payload.get("facts", [])
    if not facts:
        return {"answer": "No supporting fact found.", "confidence": 0.0}

    ranked = sorted(facts, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    top = ranked[0]
    answer = f"Best evidence for '{query}': {top.get('text', '')}"
    confidence = min(1.0, float(top.get("score", 0.0)) / max(1.0, len(str(query).split())))
    return {
        "answer": answer,
        "confidence": round(confidence, 3),
        "top_fact": top,
        "num_facts": len(facts),
    }


def _analyze_segment(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    query = str(payload.get("query", "")).strip().lower()
    text = str(payload.get("text", ""))
    segment_id = str(payload.get("segment_id", ""))
    max_facts = int(payload.get("max_facts", 6))

    query_terms = [t for t in re.findall(r"[a-zA-Z0-9_]+", query) if len(t) > 1]
    sentences = _split_sentences(text)

    facts = []
    for sentence in sentences:
        low = sentence.lower()
        score = sum(1 for term in query_terms if term in low)
        if score <= 0:
            continue
        facts.append(
            {
                "segment_id": segment_id,
                "text": sentence,
                "score": score,
            }
        )
    facts.sort(key=lambda x: x["score"], reverse=True)
    top = facts[:max_facts]

    confidence = 0.0
    if top and query_terms:
        confidence = min(1.0, float(top[0]["score"]) / max(1.0, len(query_terms)))

    return {
        "segment_id": segment_id,
        "facts": top,
        "num_facts": len(top),
        "confidence": round(confidence, 3),
    }


async def _sleep_tool(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    seconds = float(payload.get("seconds", 0.1))
    await asyncio.sleep(seconds)
    return {"slept": seconds}


def register_builtin_objects(registry: MCPRegistry) -> None:
    registry.register("extract_facts", _extract_facts)
    registry.register("merge_facts", _merge_facts)
    registry.register("analyze_segment", _analyze_segment)
    registry.register("sleep", _sleep_tool)
