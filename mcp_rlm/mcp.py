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
_MCQ_LETTERS = ("A", "B", "C", "D")


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
        except asyncio.CancelledError as exc:
            return MCPResult(
                object_name=call.object_name,
                ok=False,
                error=f"CancelledError: {exc}",
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
        async def one(call: MCPCall) -> MCPResult:
            try:
                return await self.call(call, ctx)
            except asyncio.CancelledError as exc:
                return MCPResult(object_name=call.object_name, ok=False, error=f"CancelledError: {exc}")
            except Exception as exc:
                return MCPResult(object_name=call.object_name, ok=False, error=str(exc))

        tasks = [asyncio.create_task(one(call)) for call in calls]
        return await asyncio.gather(*tasks)


def _split_sentences(text: str) -> List[str]:
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]


def _extract_answer_letter(text: str) -> Optional[str]:
    raw = str(text or "").strip().upper()
    if not raw:
        return None

    patterns = [
        r"THE\s+CORRECT\s+ANSWER\s+IS\s*\(?([A-D])\)?",
        r"ANSWER\s*[:]\s*\(?([A-D])\)?",
        r"\(([A-D])\)",
        r"\b([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            return str(match.group(1))
    return None


def _parse_mcq_choices(payload: Dict[str, Any]) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    raw_choices = payload.get("choices")
    if isinstance(raw_choices, dict):
        for key, value in raw_choices.items():
            letter = str(key).strip().upper()
            if letter in _MCQ_LETTERS:
                choices[letter] = str(value).strip()

    for letter in _MCQ_LETTERS:
        if letter in choices:
            continue
        field = f"choice_{letter}"
        if field in payload:
            choices[letter] = str(payload.get(field, "")).strip()

    return {k: v for k, v in choices.items() if v}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_choice_scores(data: Any) -> Dict[str, float]:
    out: Dict[str, float] = {k: 0.0 for k in _MCQ_LETTERS}
    if not isinstance(data, dict):
        return out
    raw = data.get("choice_scores", data)
    if not isinstance(raw, dict):
        return out
    for letter in _MCQ_LETTERS:
        out[letter] = _safe_float(raw.get(letter, 0.0), 0.0)
    return out


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", str(text).lower()) if len(t) > 1]


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


def _score_mcq_choices(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    question = str(payload.get("question") or payload.get("query") or "").strip().lower()
    text = str(payload.get("text", ""))
    segment_id = str(payload.get("segment_id", "")).strip()
    max_evidence = max(1, int(payload.get("max_evidence", 3)))

    choices = _parse_mcq_choices(payload)
    if not question or not choices or not text.strip():
        return {
            "segment_id": segment_id,
            "choice_scores": {k: 0.0 for k in _MCQ_LETTERS},
            "evidence": [],
            "best_choice": None,
            "confidence": 0.0,
        }

    question_terms = [t for t in re.findall(r"[a-zA-Z0-9_]+", question) if len(t) > 1]
    sentences = _split_sentences(text)
    per_choice_evidence: Dict[str, List[Dict[str, Any]]] = {k: [] for k in _MCQ_LETTERS}

    for letter, choice_text in choices.items():
        choice_terms = [t for t in re.findall(r"[a-zA-Z0-9_]+", choice_text.lower()) if len(t) > 1]
        for sentence in sentences:
            low = sentence.lower()
            q_overlap = sum(1 for term in question_terms if term in low)
            c_overlap = sum(1 for term in choice_terms if term in low)
            if c_overlap <= 0:
                continue
            joint = min(q_overlap, c_overlap)
            score = (0.7 * q_overlap) + (1.4 * c_overlap) + (0.8 * joint)
            if score <= 0:
                continue
            per_choice_evidence[letter].append(
                {
                    "segment_id": segment_id,
                    "choice": letter,
                    "text": sentence,
                    "score": round(float(score), 4),
                }
            )

    choice_scores: Dict[str, float] = {k: 0.0 for k in _MCQ_LETTERS}
    flattened: List[Dict[str, Any]] = []
    for letter, evidences in per_choice_evidence.items():
        evidences.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        top_hits = evidences[:max_evidence]
        choice_scores[letter] = round(sum(float(x.get("score", 0.0)) for x in top_hits), 4)
        flattened.extend(top_hits)

    ranking = sorted(choice_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_choice = ranking[0][0] if ranking and ranking[0][1] > 0 else None
    total = sum(choice_scores.values())
    confidence = 0.0 if total <= 0 else max(0.0, min(1.0, ranking[0][1] / total))

    flattened.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return {
        "segment_id": segment_id,
        "choice_scores": choice_scores,
        "ranking": [{"choice": c, "score": s} for c, s in ranking],
        "evidence": flattened[: max_evidence * max(1, len(choices))],
        "best_choice": best_choice,
        "confidence": round(confidence, 3),
    }


def _score_mcq_windows(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    question = str(payload.get("question") or payload.get("query") or "").strip()
    choices = _parse_mcq_choices(payload)
    windows = payload.get("windows", [])
    max_windows = max(1, int(payload.get("max_windows", 8)))

    if not isinstance(windows, list):
        windows = []

    all_maps: List[Dict[str, float]] = []
    all_evidence: List[Dict[str, Any]] = []

    for idx, window in enumerate(windows[:max_windows]):
        if not isinstance(window, dict):
            continue
        text = str(window.get("text", ""))
        if not text.strip():
            continue
        window_id = str(window.get("window_id", f"W{idx:05d}"))
        response = _score_mcq_choices(
            {
                "question": question,
                "choices": choices,
                "text": text,
                "segment_id": str(payload.get("segment_id", "")),
                "max_evidence": int(payload.get("max_evidence", 2)),
            },
            _,
        )
        score_map = _normalize_choice_scores(response)
        win_weight = max(0.2, _safe_float(window.get("score", 1.0), 1.0))
        scaled_map = {k: float(v) * win_weight for k, v in score_map.items()}
        all_maps.append(scaled_map)

        for ev in list(response.get("evidence", [])):
            if isinstance(ev, dict):
                item = dict(ev)
                item["window_id"] = window_id
                item["window_offset"] = int(window.get("offset", 0))
                item["window_weight"] = win_weight
                all_evidence.append(item)

    agg = {k: 0.0 for k in _MCQ_LETTERS}
    for smap in all_maps:
        for k in _MCQ_LETTERS:
            agg[k] += _safe_float(smap.get(k, 0.0), 0.0)

    ranking = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    best_choice = ranking[0][0] if ranking and ranking[0][1] > 0 else None
    total = sum(agg.values())
    confidence = 0.0 if total <= 0 else max(0.0, min(1.0, ranking[0][1] / total))

    all_evidence.sort(key=lambda x: _safe_float(x.get("score", 0.0), 0.0), reverse=True)
    return {
        "choice_scores": {k: round(float(v), 4) for k, v in agg.items()},
        "ranking": [{"choice": c, "score": round(float(s), 4)} for c, s in ranking],
        "best_choice": best_choice,
        "confidence": round(confidence, 3),
        "num_windows": len(all_maps),
        "evidence": all_evidence[: max(4, int(payload.get("max_evidence", 16)))],
    }


def _eliminate_choices(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    text = str(payload.get("text", ""))
    choices = _parse_mcq_choices(payload)
    sentences = _split_sentences(text)

    neg_cues = [" not ", " incorrect", " false", " wrong", " cannot", " except", " rather than", " instead of"]
    pos_cues = [" is ", " are ", " correct", " true", " means", " refers to"]

    penalties: Dict[str, float] = {k: 0.0 for k in _MCQ_LETTERS}
    supports: Dict[str, float] = {k: 0.0 for k in _MCQ_LETTERS}

    for letter, choice in choices.items():
        terms = _tokenize(choice)
        if not terms:
            continue
        for sentence in sentences:
            low = f" {sentence.lower()} "
            overlap = sum(1 for t in terms if t in low)
            if overlap <= 0:
                continue
            neg = sum(1 for cue in neg_cues if cue in low)
            pos = sum(1 for cue in pos_cues if cue in low)
            if neg > 0:
                penalties[letter] += float(neg * overlap)
            if pos > 0:
                supports[letter] += float(pos * overlap)

    final_penalties = {k: round(max(0.0, penalties[k] - (0.25 * supports[k])), 4) for k in _MCQ_LETTERS}
    eliminated = [k for k in _MCQ_LETTERS if final_penalties[k] >= float(payload.get("eliminate_threshold", 2.0))]

    return {
        "choice_penalties": final_penalties,
        "supports": {k: round(float(v), 4) for k, v in supports.items()},
        "eliminated": eliminated,
    }


def _extract_code_cues(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    text = str(payload.get("text", ""))
    question = str(payload.get("question") or payload.get("query") or "")
    choices = _parse_mcq_choices(payload)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        lines = _split_sentences(text)

    question_terms = _tokenize(question)
    markers = ["def ", "class ", "import ", "return ", "function ", "=>", "::", "(", ")", "{" , "}"]

    scores = {k: 0.0 for k in _MCQ_LETTERS}
    evidence: List[Dict[str, Any]] = []

    for line in lines:
        low = line.lower()
        marker_score = sum(1 for m in markers if m in low)
        if marker_score <= 0:
            continue
        q_overlap = sum(1 for t in question_terms if t in low)
        for letter, choice in choices.items():
            c_terms = _tokenize(choice)
            c_overlap = sum(1 for t in c_terms if t in low)
            if c_overlap <= 0:
                continue
            score = (1.2 * c_overlap) + (0.6 * q_overlap) + (0.4 * marker_score)
            scores[letter] += score
            evidence.append({
                "choice": letter,
                "text": line[:400],
                "score": round(float(score), 4),
                "kind": "code",
            })

    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_choice = ranking[0][0] if ranking and ranking[0][1] > 0 else None
    total = sum(scores.values())
    confidence = 0.0 if total <= 0 else max(0.0, min(1.0, ranking[0][1] / total))

    evidence.sort(key=lambda x: _safe_float(x.get("score", 0.0)), reverse=True)
    return {
        "choice_scores": {k: round(float(v), 4) for k, v in scores.items()},
        "best_choice": best_choice,
        "confidence": round(confidence, 3),
        "evidence": evidence[: max(6, int(payload.get("max_evidence", 16)))],
    }


def _extract_table_cues(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    text = str(payload.get("text", ""))
    question = str(payload.get("question") or payload.get("query") or "")
    choices = _parse_mcq_choices(payload)

    rows: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ("|" in line) or ("\t" in line) or (line.count(",") >= 2) or (line.count(";") >= 2):
            rows.append(line)

    question_terms = _tokenize(question)
    scores = {k: 0.0 for k in _MCQ_LETTERS}
    evidence: List[Dict[str, Any]] = []

    for row in rows:
        low = row.lower()
        q_overlap = sum(1 for t in question_terms if t in low)
        for letter, choice in choices.items():
            c_terms = _tokenize(choice)
            c_overlap = sum(1 for t in c_terms if t in low)
            if c_overlap <= 0:
                continue
            score = (1.3 * c_overlap) + (0.7 * q_overlap)
            scores[letter] += score
            evidence.append({
                "choice": letter,
                "text": row[:400],
                "score": round(float(score), 4),
                "kind": "table",
            })

    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_choice = ranking[0][0] if ranking and ranking[0][1] > 0 else None
    total = sum(scores.values())
    confidence = 0.0 if total <= 0 else max(0.0, min(1.0, ranking[0][1] / total))

    evidence.sort(key=lambda x: _safe_float(x.get("score", 0.0)), reverse=True)
    return {
        "choice_scores": {k: round(float(v), 4) for k, v in scores.items()},
        "best_choice": best_choice,
        "confidence": round(confidence, 3),
        "evidence": evidence[: max(6, int(payload.get("max_evidence", 16)))],
    }


def _rerank_hits_with_choices(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    query = str(payload.get("query", "")).strip()
    choices = _parse_mcq_choices(payload)
    hits = payload.get("hits", [])
    top_k = max(1, int(payload.get("top_k", 16)))

    if not isinstance(hits, list):
        hits = []

    q_terms = _tokenize(query)
    choice_terms: List[str] = []
    for _, c in choices.items():
        choice_terms.extend(_tokenize(c))

    ranked: List[Dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        preview = str(hit.get("preview", ""))
        low = preview.lower()
        base = _safe_float(hit.get("score", 0.0), 0.0)
        q_overlap = sum(1 for t in q_terms if t in low)
        c_overlap = sum(1 for t in choice_terms if t in low)
        joint = min(q_overlap, c_overlap)
        source_bonus = float(len(hit.get("sources", []))) if isinstance(hit.get("sources"), list) else 0.0

        rerank = (0.55 * base) + (0.85 * q_overlap) + (1.1 * c_overlap) + (0.6 * joint) + (0.2 * source_bonus)
        item = dict(hit)
        item["rerank_score"] = round(float(rerank), 4)
        ranked.append(item)

    ranked.sort(key=lambda x: _safe_float(x.get("rerank_score", 0.0), 0.0), reverse=True)
    return {
        "query": query,
        "top_k": top_k,
        "hits": ranked[:top_k],
        "total_candidates": len(ranked),
    }


def _vote_choice_scores(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    maps = payload.get("maps", [])
    weights = payload.get("weights", [])
    prior_scores = _normalize_choice_scores(payload.get("prior_scores", {}))
    elimination = payload.get("elimination_penalties", {})
    elimination_weight = _safe_float(payload.get("elimination_weight", 1.3), 1.3)

    if not isinstance(maps, list):
        maps = []
    if not isinstance(weights, list):
        weights = []

    score = {k: float(prior_scores[k]) for k in _MCQ_LETTERS}

    for idx, smap in enumerate(maps):
        parsed = _normalize_choice_scores(smap)
        weight = _safe_float(weights[idx], 1.0) if idx < len(weights) else 1.0
        for k in _MCQ_LETTERS:
            score[k] += float(parsed[k]) * weight

    elim_map = _normalize_choice_scores(elimination)
    for k in _MCQ_LETTERS:
        score[k] -= elimination_weight * float(elim_map[k])

    ranking = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    best_choice = ranking[0][0] if ranking else None

    clipped = [max(0.0, s) for _, s in ranking]
    total = sum(clipped)
    confidence = 0.0 if total <= 0 else max(0.0, min(1.0, clipped[0] / total))

    return {
        "choice_scores": {k: round(float(v), 4) for k, v in score.items()},
        "ranking": [{"choice": c, "score": round(float(s), 4)} for c, s in ranking],
        "best_choice": best_choice,
        "confidence": round(confidence, 3),
        "elimination_weight": elimination_weight,
    }


def _aggregate_mcq_scores(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    items = payload.get("items", [])
    max_evidence = max(1, int(payload.get("max_evidence", 24)))

    choice_scores: Dict[str, float] = {k: 0.0 for k in _MCQ_LETTERS}
    all_evidence: List[Dict[str, Any]] = []

    if not isinstance(items, list):
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        weight = float(item.get("weight", 1.0))
        raw_scores = item.get("choice_scores", {})
        if isinstance(raw_scores, dict):
            for letter in _MCQ_LETTERS:
                try:
                    choice_scores[letter] += float(raw_scores.get(letter, 0.0)) * weight
                except (TypeError, ValueError):
                    continue
        raw_evidence = item.get("evidence", [])
        if isinstance(raw_evidence, list):
            for raw in raw_evidence:
                if isinstance(raw, dict):
                    all_evidence.append(raw)

    for letter in _MCQ_LETTERS:
        choice_scores[letter] = round(choice_scores[letter], 4)

    ranking = sorted(choice_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_choice = ranking[0][0] if ranking and ranking[0][1] > 0 else None
    total = sum(choice_scores.values())
    confidence = 0.0 if total <= 0 else max(0.0, min(1.0, ranking[0][1] / total))

    all_evidence.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return {
        "choice_scores": choice_scores,
        "ranking": [{"choice": c, "score": s} for c, s in ranking],
        "best_choice": best_choice,
        "confidence": round(confidence, 3),
        "evidence": all_evidence[:max_evidence],
    }


def _normalize_mcq_answer(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    response = str(payload.get("response", "")).strip()
    fallback = str(payload.get("fallback", "")).strip().upper()

    answer = _extract_answer_letter(response)
    used_fallback = False
    if answer is None and fallback in _MCQ_LETTERS:
        answer = fallback
        used_fallback = True

    normalized = ""
    if answer in _MCQ_LETTERS:
        normalized = f"The correct answer is ({answer})"

    return {
        "answer": answer,
        "normalized_response": normalized,
        "used_fallback": used_fallback,
        "is_valid": answer in _MCQ_LETTERS,
    }


async def _sleep_tool(payload: Dict[str, Any], _: MCPInvocationContext) -> Dict[str, Any]:
    seconds = float(payload.get("seconds", 0.1))
    await asyncio.sleep(seconds)
    return {"slept": seconds}


def register_builtin_objects(registry: MCPRegistry) -> None:
    registry.register("extract_facts", _extract_facts)
    registry.register("merge_facts", _merge_facts)
    registry.register("analyze_segment", _analyze_segment)

    # LongBench-v2 score-oriented objects
    registry.register("score_mcq_choices", _score_mcq_choices)
    registry.register("score_mcq_windows", _score_mcq_windows)
    registry.register("eliminate_choices", _eliminate_choices)
    registry.register("extract_code_cues", _extract_code_cues)
    registry.register("extract_table_cues", _extract_table_cues)
    registry.register("vote_choice_scores", _vote_choice_scores)
    registry.register("rerank_hits_with_choices", _rerank_hits_with_choices)
    registry.register("aggregate_mcq_scores", _aggregate_mcq_scores)
    registry.register("normalize_mcq_answer", _normalize_mcq_answer)

    registry.register("sleep", _sleep_tool)
