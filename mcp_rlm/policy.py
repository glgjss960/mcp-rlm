from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from time import perf_counter


def _to_bool(raw: Any, *, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_timeout(raw: Any, *, default: float, low: float = 0.1, high: float = 7200.0) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _stage_log(
    *,
    enabled: bool,
    component: str,
    stage: str,
    status: str,
    elapsed_ms: Optional[int] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    if not enabled:
        return
    payload: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "stage": stage,
        "status": status,
    }
    if elapsed_ms is not None:
        payload["elapsed_ms"] = int(elapsed_ms)
    if detail:
        payload["detail"] = detail
    print("[mcp-rlm-stage] " + json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


class ModelJSONParseError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        raw_text: str = "",
        candidate_count: int = 0,
        last_error: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.candidate_count = int(candidate_count)
        self.last_error = str(last_error or "")


def _strip_markdown_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return cleaned
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _collect_json_object_candidates(text: str) -> List[str]:
    payload = _strip_markdown_fence(text)
    out: List[str] = []
    start = -1
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(payload):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "\"":
                in_string = False
            continue

        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(payload[start : idx + 1])
                start = -1
    if out:
        return out

    # Legacy fallback: from first '{' to last '}'.
    first = payload.find("{")
    last = payload.rfind("}")
    if first >= 0 and last > first:
        return [payload[first : last + 1]]
    return []


def _repair_multiline_missing_commas(raw: str) -> str:
    lines = str(raw or "").splitlines()
    if len(lines) <= 2:
        return str(raw or "")
    fixed = list(lines)
    for idx in range(1, len(fixed)):
        prev = fixed[idx - 1].rstrip()
        curr = fixed[idx].lstrip()
        if not prev or not curr.startswith("\""):
            continue
        if prev.endswith((",", "{", "[", ":")):
            continue
        fixed[idx - 1] = prev + ","
    return "\n".join(fixed)


def _repair_json_text(raw: str) -> str:
    text = _strip_markdown_fence(raw)
    if not text:
        return text
    # Remove trailing commas before } or ].
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Repair common missing-comma failure in multiline key/value JSON.
    text = _repair_multiline_missing_commas(text)
    return text


def extract_json_object_from_text(text: str, *, allow_repair: bool = False) -> Dict[str, Any]:
    raw_text = _strip_markdown_fence(text)
    candidates = _collect_json_object_candidates(raw_text)
    if not candidates:
        raise ModelJSONParseError("No JSON object found in model output", raw_text=raw_text)

    parse_errors: List[str] = []
    for candidate in candidates:
        attempts = [candidate]
        if allow_repair:
            repaired = _repair_json_text(candidate)
            if repaired and repaired != candidate:
                attempts.append(repaired)
        for current in attempts:
            try:
                parsed = json.loads(current)
            except Exception as exc:
                parse_errors.append(str(exc))
                continue
            if isinstance(parsed, dict):
                return parsed
            parse_errors.append("JSON root must be object")

    last_error = parse_errors[-1] if parse_errors else ""
    raise ModelJSONParseError(
        "Failed to parse JSON object from model output",
        raw_text=raw_text,
        candidate_count=len(candidates),
        last_error=last_error,
    )


def _extract_json_object(text: str, *, allow_repair: bool = False) -> Dict[str, Any]:
    return extract_json_object_from_text(text, allow_repair=allow_repair)


@dataclass
class SearchPlan:
    top_k: int
    coarse_k: int
    max_children: int
    read_max_chars: int


class BasePolicy:
    async def plan_root(self, *, query: str, context_stats: Dict[str, Any]) -> SearchPlan:
        raise NotImplementedError

    async def leaf_instruction(self, *, query: str, segment_meta: Dict[str, Any]) -> str:
        raise NotImplementedError

    async def finalize_answer(self, *, query: str, merged: Dict[str, Any], facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError


def _clamp_int(value: Any, *, low: int, high: int, default: int) -> int:
    try:
        raw = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, raw))


class HeuristicPolicy(BasePolicy):
    async def plan_root(self, *, query: str, context_stats: Dict[str, Any]) -> SearchPlan:
        leaf_segments = int(context_stats.get("leaf_segments", 1))
        if leaf_segments <= 16:
            return SearchPlan(top_k=min(leaf_segments, 8), coarse_k=8, max_children=8, read_max_chars=12000)
        if leaf_segments <= 64:
            return SearchPlan(top_k=12, coarse_k=20, max_children=12, read_max_chars=12000)
        return SearchPlan(top_k=16, coarse_k=32, max_children=16, read_max_chars=10000)

    async def leaf_instruction(self, *, query: str, segment_meta: Dict[str, Any]) -> str:
        return (
            "Extract concise evidence strictly relevant to the query. "
            "Prefer exact strings, entities, numbers, and short factual spans."
        )

    async def finalize_answer(self, *, query: str, merged: Dict[str, Any], facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        answer = str(merged.get("answer", "No answer."))
        confidence = float(merged.get("confidence", 0.0))
        return {
            "answer": answer,
            "confidence": confidence,
            "num_facts": len(facts),
            "policy": "heuristic",
        }


class OpenAICompatiblePolicy(BasePolicy):
    def __init__(
        self,
        *,
        provider_mode: str,
        api_base: str,
        model: str,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 25.0,
        fallback: Optional[BasePolicy] = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.provider_mode = str(provider_mode or "").strip().lower()
        self.api_key = api_key
        self.extra_headers = dict(extra_headers or {})
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback or HeuristicPolicy()

    async def plan_root(self, *, query: str, context_stats: Dict[str, Any]) -> SearchPlan:
        prompt = {
            "task": "Create search/branching plan for hierarchical long-context QA.",
            "query": query,
            "context_stats": context_stats,
            "output_schema": {
                "top_k": "int in [4,32]",
                "coarse_k": "int in [8,64]",
                "max_children": "int in [4,32]",
                "read_max_chars": "int in [4000,24000]",
            },
        }
        try:
            result = await self._chat_json(system="You are a planning policy for recursive agents.", user=json.dumps(prompt, ensure_ascii=False))
            return SearchPlan(
                top_k=_clamp_int(result.get("top_k"), low=4, high=32, default=12),
                coarse_k=_clamp_int(result.get("coarse_k"), low=8, high=64, default=24),
                max_children=_clamp_int(result.get("max_children"), low=4, high=32, default=12),
                read_max_chars=_clamp_int(result.get("read_max_chars"), low=4000, high=24000, default=12000),
            )
        except Exception:
            return await self.fallback.plan_root(query=query, context_stats=context_stats)

    async def leaf_instruction(self, *, query: str, segment_meta: Dict[str, Any]) -> str:
        prompt = {
            "task": "Produce extraction instruction for a leaf segment.",
            "query": query,
            "segment_meta": segment_meta,
            "output_schema": {"instruction": "string"},
        }
        try:
            result = await self._chat_json(system="You are an extraction planner.", user=json.dumps(prompt, ensure_ascii=False))
            text = str(result.get("instruction", "")).strip()
            if text:
                return text
        except Exception:
            pass
        return await self.fallback.leaf_instruction(query=query, segment_meta=segment_meta)

    async def finalize_answer(self, *, query: str, merged: Dict[str, Any], facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = {
            "task": "Refine final answer with confidence.",
            "query": query,
            "merged": merged,
            "facts": facts[:20],
            "output_schema": {
                "answer": "string",
                "confidence": "float in [0,1]",
            },
        }
        try:
            result = await self._chat_json(system="You are a final answer policy.", user=json.dumps(prompt, ensure_ascii=False))
            answer = str(result.get("answer", merged.get("answer", "")))
            confidence = float(result.get("confidence", merged.get("confidence", 0.0)))
            return {
                "answer": answer,
                "confidence": max(0.0, min(1.0, confidence)),
                "num_facts": len(facts),
                "policy": f"openai:{self.model}",
            }
        except Exception:
            return await self.fallback.finalize_answer(query=query, merged=merged, facts=facts)

    async def _chat_json(
        self,
        *,
        system: str,
        user: str,
        max_new_tokens: Optional[int] = None,
        json_mode: str = "none",
        json_schema: Optional[Dict[str, Any]] = None,
        json_repair: bool = False,
        return_meta: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system + " Return ONLY valid JSON object."},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        if max_new_tokens is not None:
            payload["max_tokens"] = max(8, int(max_new_tokens))

        normalized_mode = str(json_mode or "none").strip().lower()
        if normalized_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
            if self.provider_mode == "vllm":
                payload["guided_json"] = {"type": "object"}
        elif normalized_mode == "json_schema":
            schema = json_schema if isinstance(json_schema, dict) else {"type": "object"}
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "mcp_rlm_response",
                    "schema": schema,
                },
            }
            if self.provider_mode == "vllm":
                payload["guided_json"] = schema

        def _request(payload_obj: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
            data = json.dumps(payload_obj).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            headers.update(self.extra_headers)

            req = urllib.request.Request(
                url=f"{self.api_base}/chat/completions",
                data=data,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
            raw = json.loads(body)
            content = raw["choices"][0]["message"]["content"]
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text is not None:
                            parts.append(str(text))
                    elif item is not None:
                        parts.append(str(item))
                content = "".join(parts)
            parsed = _extract_json_object(str(content), allow_repair=json_repair)
            return parsed, str(content)
        try:
            parsed, raw_text = await asyncio.to_thread(_request, payload)
            if return_meta:
                return {"_mcp_rlm_parsed": parsed, "_mcp_rlm_raw_text": raw_text}
            return parsed
        except Exception as exc:
            # Some OpenAI-compatible providers reject response_format=json_schema/json_object.
            if "response_format" in payload and not isinstance(exc, ModelJSONParseError):
                fallback_payload = dict(payload)
                fallback_payload.pop("response_format", None)
                fallback_payload.pop("guided_json", None)
                parsed, raw_text = await asyncio.to_thread(_request, fallback_payload)
                if return_meta:
                    return {"_mcp_rlm_parsed": parsed, "_mcp_rlm_raw_text": raw_text}
                return parsed
            raise


class TransformersLocalPolicy(BasePolicy):
    """Optional local policy via `transformers` text-generation pipeline."""

    def __init__(
        self,
        *,
        model: str,
        revision: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        chat_timeout_seconds: float = 120.0,
        load_timeout_seconds: float = 1800.0,
        generate_timeout_seconds: float = 120.0,
        stage_log_enabled: bool = True,
        use_worker_process: bool = True,
        worker_module: str = "mcp_rlm.hf_worker",
        fallback: Optional[BasePolicy] = None,
    ) -> None:
        self.model = model
        self.revision = revision
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.chat_timeout_seconds = _to_timeout(chat_timeout_seconds, default=120.0)
        self.load_timeout_seconds = _to_timeout(
            load_timeout_seconds,
            default=max(600.0, self.chat_timeout_seconds),
            low=1.0,
        )
        self.generate_timeout_seconds = _to_timeout(
            generate_timeout_seconds,
            default=self.chat_timeout_seconds,
        )
        self.stage_log_enabled = bool(stage_log_enabled)
        self.use_worker_process = bool(use_worker_process)
        self.worker_module = str(worker_module or "mcp_rlm.hf_worker").strip() or "mcp_rlm.hf_worker"
        self.fallback = fallback or HeuristicPolicy()

        self._pipeline: Any = None
        self._load_lock = asyncio.Lock()
        self._worker_proc: Optional[asyncio.subprocess.Process] = None
        self._worker_lock = asyncio.Lock()
        self._worker_stderr_task: Optional[asyncio.Task[Any]] = None
        self._worker_seq = 0

    async def plan_root(self, *, query: str, context_stats: Dict[str, Any]) -> SearchPlan:
        prompt = {
            "task": "Create search/branching plan for hierarchical long-context QA.",
            "query": query,
            "context_stats": context_stats,
            "output_schema": {
                "top_k": "int in [4,32]",
                "coarse_k": "int in [8,64]",
                "max_children": "int in [4,32]",
                "read_max_chars": "int in [4000,24000]",
            },
        }
        try:
            result = await self._chat_json(system="You are a planning policy for recursive agents.", user=json.dumps(prompt, ensure_ascii=False))
            return SearchPlan(
                top_k=_clamp_int(result.get("top_k"), low=4, high=32, default=12),
                coarse_k=_clamp_int(result.get("coarse_k"), low=8, high=64, default=24),
                max_children=_clamp_int(result.get("max_children"), low=4, high=32, default=12),
                read_max_chars=_clamp_int(result.get("read_max_chars"), low=4000, high=24000, default=12000),
            )
        except Exception:
            return await self.fallback.plan_root(query=query, context_stats=context_stats)

    async def leaf_instruction(self, *, query: str, segment_meta: Dict[str, Any]) -> str:
        prompt = {
            "task": "Produce extraction instruction for a leaf segment.",
            "query": query,
            "segment_meta": segment_meta,
            "output_schema": {"instruction": "string"},
        }
        try:
            result = await self._chat_json(system="You are an extraction planner.", user=json.dumps(prompt, ensure_ascii=False))
            text = str(result.get("instruction", "")).strip()
            if text:
                return text
        except Exception:
            pass
        return await self.fallback.leaf_instruction(query=query, segment_meta=segment_meta)

    async def finalize_answer(self, *, query: str, merged: Dict[str, Any], facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = {
            "task": "Refine final answer with confidence.",
            "query": query,
            "merged": merged,
            "facts": facts[:20],
            "output_schema": {
                "answer": "string",
                "confidence": "float in [0,1]",
            },
        }
        try:
            result = await self._chat_json(system="You are a final answer policy.", user=json.dumps(prompt, ensure_ascii=False))
            answer = str(result.get("answer", merged.get("answer", "")))
            confidence = float(result.get("confidence", merged.get("confidence", 0.0)))
            return {
                "answer": answer,
                "confidence": max(0.0, min(1.0, confidence)),
                "num_facts": len(facts),
                "policy": f"hf:{self.model}",
            }
        except Exception:
            return await self.fallback.finalize_answer(query=query, merged=merged, facts=facts)

    async def _chat_json(
        self,
        *,
        system: str,
        user: str,
        max_new_tokens: Optional[int] = None,
        json_mode: str = "none",
        json_schema: Optional[Dict[str, Any]] = None,
        json_repair: bool = False,
        return_meta: bool = False,
    ) -> Dict[str, Any]:
        start = perf_counter()
        _stage_log(
            enabled=self.stage_log_enabled,
            component="policy.hf",
            stage="chat_json",
            status="start",
            detail={
                "model": self.model,
                "chat_timeout_seconds": self.chat_timeout_seconds,
                "load_timeout_seconds": self.load_timeout_seconds,
                "generate_timeout_seconds": self.generate_timeout_seconds,
                "user_chars": len(user),
                "system_chars": len(system),
                "json_mode": str(json_mode or "none"),
            },
        )

        try:
            _stage_log(
                enabled=self.stage_log_enabled,
                component="policy.hf",
                stage="chat_json.ensure_loaded",
                status="start",
                detail={"model": self.model, "timeout_seconds": self.load_timeout_seconds},
            )
            load_start = perf_counter()
            try:
                await asyncio.wait_for(
                    self._ensure_loaded(),
                    timeout=self.load_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                _stage_log(
                    enabled=self.stage_log_enabled,
                    component="policy.hf",
                    stage="chat_json.ensure_loaded",
                    status="timeout",
                    elapsed_ms=int((perf_counter() - load_start) * 1000),
                    detail={"model": self.model, "timeout_seconds": self.load_timeout_seconds},
                )
                raise RuntimeError(
                    f"HF ensure_loaded timed out after {self.load_timeout_seconds:.1f}s for model={self.model}"
                ) from exc
            except Exception as exc:
                _stage_log(
                    enabled=self.stage_log_enabled,
                    component="policy.hf",
                    stage="chat_json.ensure_loaded",
                    status="error",
                    elapsed_ms=int((perf_counter() - load_start) * 1000),
                    detail={"model": self.model, "error": str(exc), "error_type": type(exc).__name__},
                )
                raise
            _stage_log(
                enabled=self.stage_log_enabled,
                component="policy.hf",
                stage="chat_json.ensure_loaded",
                status="ok",
                elapsed_ms=int((perf_counter() - load_start) * 1000),
                detail={"model": self.model},
            )

            generation_tokens = max(8, int(max_new_tokens if max_new_tokens is not None else self.max_new_tokens))
            _stage_log(
                enabled=self.stage_log_enabled,
                component="policy.hf",
                stage="chat_json.generate",
                status="start",
                detail={
                    "model": self.model,
                    "timeout_seconds": self.generate_timeout_seconds,
                    "max_new_tokens": generation_tokens,
                    "use_worker_process": self.use_worker_process,
                },
            )
            generate_start = perf_counter()
            try:
                if self.use_worker_process:
                    text = await self._generate_with_worker(
                        system=system,
                        user=user,
                        max_new_tokens=generation_tokens,
                        json_mode=json_mode,
                        json_schema=json_schema,
                    )
                else:
                    messages = [
                        {"role": "system", "content": system + " Return ONLY valid JSON object."},
                        {"role": "user", "content": user},
                    ]
                    _stage_log(
                        enabled=self.stage_log_enabled,
                        component="policy.hf",
                        stage="chat_json.render_prompt",
                        status="start",
                        detail={"model": self.model},
                    )
                    render_start = perf_counter()
                    prompt = await asyncio.to_thread(self._chat_prompt_from_messages, messages)
                    _stage_log(
                        enabled=self.stage_log_enabled,
                        component="policy.hf",
                        stage="chat_json.render_prompt",
                        status="ok",
                        elapsed_ms=int((perf_counter() - render_start) * 1000),
                        detail={"model": self.model, "prompt_chars": len(prompt)},
                    )

                    def _generate_text(rendered_prompt: str) -> str:
                        if self._pipeline is None:
                            raise RuntimeError("Transformers pipeline is not initialized")
                        try:
                            outputs = self._pipeline(
                                rendered_prompt,
                                max_new_tokens=generation_tokens,
                                do_sample=False,
                                temperature=0.0,
                                return_full_text=False,
                            )
                        except TypeError:
                            outputs = self._pipeline(
                                rendered_prompt,
                                max_new_tokens=generation_tokens,
                                do_sample=False,
                                temperature=0.0,
                            )
                        if not outputs:
                            raise RuntimeError("Empty generation output")
                        first = outputs[0]
                        return self._extract_generation_text(first, prompt=rendered_prompt)

                    text = await asyncio.wait_for(
                        asyncio.to_thread(_generate_text, prompt),
                        timeout=self.generate_timeout_seconds,
                    )
            except asyncio.TimeoutError as exc:
                _stage_log(
                    enabled=self.stage_log_enabled,
                    component="policy.hf",
                    stage="chat_json.generate",
                    status="timeout",
                    elapsed_ms=int((perf_counter() - generate_start) * 1000),
                    detail={
                        "model": self.model,
                        "timeout_seconds": self.generate_timeout_seconds,
                        "note": (
                            "subprocess worker restarted after timeout"
                            if self.use_worker_process
                            else "to_thread task may continue in background until generation returns"
                        ),
                    },
                )
                raise RuntimeError(
                    f"HF generate timed out after {self.generate_timeout_seconds:.1f}s for model={self.model}"
                ) from exc
            except Exception as exc:
                _stage_log(
                    enabled=self.stage_log_enabled,
                    component="policy.hf",
                    stage="chat_json.generate",
                    status="error",
                    elapsed_ms=int((perf_counter() - generate_start) * 1000),
                    detail={"model": self.model, "error": str(exc), "error_type": type(exc).__name__},
                )
                raise
            _stage_log(
                enabled=self.stage_log_enabled,
                component="policy.hf",
                stage="chat_json.generate",
                status="ok",
                elapsed_ms=int((perf_counter() - generate_start) * 1000),
                detail={"model": self.model, "text_chars": len(text)},
            )

            _stage_log(
                enabled=self.stage_log_enabled,
                component="policy.hf",
                stage="chat_json.parse_json",
                status="start",
                detail={"model": self.model},
            )
            parse_start = perf_counter()
            try:
                result = _extract_json_object(text, allow_repair=json_repair)
            except Exception as exc:
                _stage_log(
                    enabled=self.stage_log_enabled,
                    component="policy.hf",
                    stage="chat_json.parse_json",
                    status="error",
                    elapsed_ms=int((perf_counter() - parse_start) * 1000),
                    detail={
                        "model": self.model,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "text_preview": str(text)[:400],
                    },
                )
                raise
            _stage_log(
                enabled=self.stage_log_enabled,
                component="policy.hf",
                stage="chat_json.parse_json",
                status="ok",
                elapsed_ms=int((perf_counter() - parse_start) * 1000),
                detail={"model": self.model},
            )
        except Exception as exc:
            elapsed_ms = int((perf_counter() - start) * 1000)
            _stage_log(
                enabled=self.stage_log_enabled,
                component="policy.hf",
                stage="chat_json",
                status="error",
                elapsed_ms=elapsed_ms,
                detail={
                    "model": self.model,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise

        elapsed_ms = int((perf_counter() - start) * 1000)
        _stage_log(
            enabled=self.stage_log_enabled,
            component="policy.hf",
            stage="chat_json",
            status="ok",
            elapsed_ms=elapsed_ms,
            detail={"model": self.model},
        )
        if return_meta:
            return {"_mcp_rlm_parsed": result, "_mcp_rlm_raw_text": str(text)}
        return result

    def _chat_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        if self._pipeline is None:
            raise RuntimeError("Transformers pipeline is not initialized")
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is not None:
            apply_template = getattr(tokenizer, "apply_chat_template", None)
            if callable(apply_template):
                try:
                    rendered = apply_template(messages, tokenize=False, add_generation_prompt=True)
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered
                except TypeError:
                    try:
                        rendered = apply_template(messages, tokenize=False)
                        if isinstance(rendered, str) and rendered.strip():
                            return rendered
                    except Exception:
                        pass
                except Exception:
                    pass

        return (
            "System:\n"
            + str(messages[0].get("content", ""))
            + "\n\nUser:\n"
            + str(messages[1].get("content", ""))
            + "\n\nAssistant:\n"
        )
    @staticmethod
    def _extract_generation_text(first: Any, *, prompt: str) -> str:
        if isinstance(first, dict):
            generated = first.get("generated_text", "")
            if isinstance(generated, str):
                text = generated
            elif isinstance(generated, list):
                candidate = ""
                for item in generated:
                    if not isinstance(item, dict):
                        continue
                    role = str(item.get("role", "")).strip().lower()
                    content = str(item.get("content", ""))
                    if role == "assistant" and content.strip():
                        candidate = content
                text = candidate or str(generated)
            else:
                text = str(generated)

            if not text and first.get("text") is not None:
                text = str(first.get("text", ""))
        else:
            text = str(first)

        if text.startswith(prompt):
            return text[len(prompt) :]
        return text

    async def _ensure_loaded(self) -> None:
        if self.use_worker_process:
            if self._worker_proc is not None and self._worker_proc.returncode is None:
                return
            async with self._load_lock:
                if self._worker_proc is not None and self._worker_proc.returncode is None:
                    return
                await self._start_worker()
            return

        if self._pipeline is not None:
            return

        async with self._load_lock:
            if self._pipeline is not None:
                return

            def _load() -> Any:
                try:
                    from transformers import pipeline
                except Exception as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError("transformers package is required for huggingface policy mode") from exc

                kwargs: Dict[str, Any] = {}
                if self.revision:
                    kwargs["revision"] = self.revision
                if self.device_map:
                    kwargs["device_map"] = self.device_map
                if self.torch_dtype and self.torch_dtype != "auto":
                    kwargs["torch_dtype"] = self.torch_dtype

                try:
                    return pipeline("text-generation", model=self.model, tokenizer=self.model, **kwargs)
                except TypeError:
                    kwargs.pop("device_map", None)
                    kwargs.pop("torch_dtype", None)
                    return pipeline("text-generation", model=self.model, tokenizer=self.model, **kwargs)

            self._pipeline = await asyncio.to_thread(_load)

    async def _start_worker(self) -> None:
        await self._stop_worker()
        cmd = [sys.executable, "-m", self.worker_module, "--model", self.model]
        if self.revision:
            cmd.extend(["--revision", self.revision])
        if self.device_map:
            cmd.extend(["--device-map", self.device_map])
        if self.torch_dtype:
            cmd.extend(["--torch-dtype", self.torch_dtype])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._worker_proc = proc
        if proc.stderr is not None:
            self._worker_stderr_task = asyncio.create_task(self._drain_worker_stderr(proc))

        if proc.stdout is None:
            await self._stop_worker()
            raise RuntimeError("HF worker failed to create stdout pipe")

        try:
            ready_raw = await asyncio.wait_for(proc.stdout.readline(), timeout=self.load_timeout_seconds)
        except asyncio.TimeoutError as exc:
            await self._stop_worker()
            raise RuntimeError(
                f"HF ensure_loaded timed out after {self.load_timeout_seconds:.1f}s for model={self.model}"
            ) from exc
        if not ready_raw:
            await self._stop_worker()
            raise RuntimeError("HF worker exited before ready handshake")
        try:
            ready = json.loads(ready_raw.decode("utf-8", errors="replace"))
        except Exception as exc:
            await self._stop_worker()
            raise RuntimeError(f"HF worker ready parse failed: {exc}") from exc
        if not bool(ready.get("ok", False)):
            error = str(ready.get("error", "unknown startup error"))
            await self._stop_worker()
            raise RuntimeError(f"HF worker startup failed: {error}")

    async def _stop_worker(self) -> None:
        proc = self._worker_proc
        self._worker_proc = None
        stderr_task = self._worker_stderr_task
        self._worker_stderr_task = None
        if stderr_task is not None:
            stderr_task.cancel()
        if proc is None:
            return
        try:
            if proc.returncode is None and proc.stdin is not None:
                try:
                    proc.stdin.write((json.dumps({"type": "shutdown"}) + "\n").encode("utf-8"))
                    await proc.stdin.drain()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except Exception:
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                    except Exception:
                        proc.kill()
                        await proc.wait()
        except Exception:
            return

    async def _restart_worker(self) -> None:
        await self._stop_worker()
        await self._start_worker()

    async def _drain_worker_stderr(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stderr is None:
            return
        try:
            while True:
                raw = await proc.stderr.readline()
                if not raw:
                    return
                _stage_log(
                    enabled=self.stage_log_enabled,
                    component="policy.hf",
                    stage="worker.stderr",
                    status="line",
                    detail={"line": raw.decode("utf-8", errors="replace").strip()[:1000]},
                )
        except Exception:
            return

    async def _generate_with_worker(
        self,
        *,
        system: str,
        user: str,
        max_new_tokens: int,
        json_mode: str,
        json_schema: Optional[Dict[str, Any]],
    ) -> str:
        async with self._worker_lock:
            await self._ensure_loaded()
            proc = self._worker_proc
            if proc is None or proc.stdin is None or proc.stdout is None:
                raise RuntimeError("HF worker is not available")
            self._worker_seq += 1
            req_id = f"req_{self._worker_seq}"
            req = {
                "id": req_id,
                "type": "chat_json",
                "system": system,
                "user": user,
                "max_new_tokens": max_new_tokens,
                "json_mode": str(json_mode or "none"),
                "json_schema": json_schema if isinstance(json_schema, dict) else None,
            }
            try:
                proc.stdin.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
                await asyncio.wait_for(proc.stdin.drain(), timeout=3.0)
            except Exception as exc:
                await self._restart_worker()
                raise RuntimeError(f"HF worker write failed: {exc}") from exc

            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=self.generate_timeout_seconds)
            except asyncio.TimeoutError as exc:
                await self._restart_worker()
                raise RuntimeError(
                    f"HF generate timed out after {self.generate_timeout_seconds:.1f}s for model={self.model}"
                ) from exc
            except Exception as exc:
                await self._restart_worker()
                raise RuntimeError(f"HF worker read failed: {exc}") from exc
            if not raw:
                await self._restart_worker()
                raise RuntimeError("HF worker exited during generation")

            try:
                payload = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception as exc:
                await self._restart_worker()
                raise RuntimeError(f"HF worker response parse failed: {exc}") from exc

            if str(payload.get("id", "")) != req_id:
                await self._restart_worker()
                raise RuntimeError("HF worker response id mismatch")

            if not bool(payload.get("ok", False)):
                raise RuntimeError(str(payload.get("error", "HF worker generation failed")))
            return str(payload.get("text", ""))


def _normalize_policy_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "mode": os.getenv("MCP_RLM_POLICY_MODE", "heuristic").strip().lower(),
        "api_base": os.getenv("MCP_RLM_API_BASE", "").strip(),
        "model": os.getenv("MCP_RLM_MODEL", "").strip(),
        "api_key": os.getenv("MCP_RLM_API_KEY", "").strip(),
        "openrouter_site_url": os.getenv("OPENROUTER_SITE_URL", "").strip(),
        "openrouter_app_name": os.getenv("OPENROUTER_APP_NAME", "").strip(),
        "hf_revision": os.getenv("MCP_RLM_HF_REVISION", "").strip(),
        "hf_device_map": os.getenv("MCP_RLM_HF_DEVICE_MAP", "auto").strip() or "auto",
        "hf_torch_dtype": os.getenv("MCP_RLM_HF_TORCH_DTYPE", "auto").strip() or "auto",
        "hf_max_new_tokens": os.getenv("MCP_RLM_HF_MAX_NEW_TOKENS", "").strip(),
        "hf_chat_timeout_seconds": os.getenv("MCP_RLM_HF_CHAT_TIMEOUT_SECONDS", "120").strip(),
        "hf_load_timeout_seconds": os.getenv("MCP_RLM_HF_LOAD_TIMEOUT_SECONDS", "1800").strip(),
        "hf_generate_timeout_seconds": os.getenv("MCP_RLM_HF_GENERATE_TIMEOUT_SECONDS", "120").strip(),
        "hf_stage_logs": os.getenv("MCP_RLM_DEBUG_STAGE_LOGS", "1").strip(),
        "hf_use_worker_process": os.getenv("MCP_RLM_HF_USE_WORKER_PROCESS", "1").strip(),
        "hf_worker_module": os.getenv("MCP_RLM_HF_WORKER_MODULE", "mcp_rlm.hf_worker").strip(),
        "request_timeout_seconds": os.getenv("MCP_RLM_REQUEST_TIMEOUT_SECONDS", "25").strip(),
    }
    if config:
        for key, value in config.items():
            if value is None:
                continue
            merged[str(key)] = value

    mode = str(merged.get("mode", "heuristic")).strip().lower()
    merged["mode"] = mode

    if mode == "openrouter":
        merged["api_base"] = str(merged.get("api_base") or "https://openrouter.ai/api/v1").strip()
        merged["api_key"] = str(merged.get("api_key") or os.getenv("OPENROUTER_API_KEY", "")).strip()
    elif mode == "openai":
        merged["api_base"] = str(merged.get("api_base") or "https://api.openai.com/v1").strip()
        merged["api_key"] = str(merged.get("api_key") or os.getenv("OPENAI_API_KEY", "")).strip()
    elif mode == "vllm":
        merged["api_base"] = str(merged.get("api_base") or "http://127.0.0.1:8000/v1").strip()
    elif mode == "ollama":
        merged["api_base"] = str(merged.get("api_base") or "http://127.0.0.1:11434/v1").strip()

    return merged


_POLICY_CACHE: Dict[str, BasePolicy] = {}


def build_policy_from_config(config: Optional[Dict[str, Any]] = None) -> BasePolicy:
    resolved = _normalize_policy_config(config)
    cache_key = json.dumps(resolved, ensure_ascii=False, sort_keys=True, default=str)
    cached = _POLICY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mode = str(resolved.get("mode", "heuristic")).strip().lower()
    fallback = HeuristicPolicy()

    if mode in {"heuristic", "rule", "rules"}:
        policy: BasePolicy = fallback
    elif mode in {"openai", "openrouter", "vllm", "ollama", "openai_compatible"}:
        api_base = str(resolved.get("api_base", "")).strip()
        model = str(resolved.get("model", "")).strip()
        if not api_base or not model:
            policy = fallback
        else:
            headers: Dict[str, str] = {}
            if mode == "openrouter":
                site = str(resolved.get("openrouter_site_url", "")).strip()
                app = str(resolved.get("openrouter_app_name", "")).strip()
                if site:
                    headers["HTTP-Referer"] = site
                if app:
                    headers["X-Title"] = app
            api_key = str(resolved.get("api_key", "")).strip() or None
            policy = OpenAICompatiblePolicy(
                provider_mode=mode,
                api_base=api_base,
                model=model,
                api_key=api_key,
                extra_headers=headers,
                timeout_seconds=_to_timeout(
                    resolved.get("request_timeout_seconds"),
                    default=25.0,
                ),
                fallback=fallback,
            )
    elif mode in {"huggingface", "hf", "transformers"}:
        model = str(resolved.get("model", "")).strip()
        if not model:
            policy = fallback
        else:
            chat_timeout_seconds = _to_timeout(
                resolved.get("hf_chat_timeout_seconds"),
                default=120.0,
            )
            policy = TransformersLocalPolicy(
                model=model,
                revision=str(resolved.get("hf_revision", "")).strip() or None,
                device_map=str(resolved.get("hf_device_map", "auto")).strip() or "auto",
                torch_dtype=str(resolved.get("hf_torch_dtype", "auto")).strip() or "auto",
                max_new_tokens=_clamp_int(
                    resolved.get("hf_max_new_tokens"),
                    low=32,
                    high=2048,
                    default=256,
                ),
                chat_timeout_seconds=chat_timeout_seconds,
                load_timeout_seconds=_to_timeout(
                    resolved.get("hf_load_timeout_seconds"),
                    default=max(600.0, chat_timeout_seconds),
                ),
                generate_timeout_seconds=_to_timeout(
                    resolved.get("hf_generate_timeout_seconds"),
                    default=chat_timeout_seconds,
                ),
                stage_log_enabled=_to_bool(resolved.get("hf_stage_logs"), default=True),
                use_worker_process=_to_bool(resolved.get("hf_use_worker_process"), default=True),
                worker_module=str(resolved.get("hf_worker_module", "mcp_rlm.hf_worker")).strip() or "mcp_rlm.hf_worker",
                fallback=fallback,
            )
    else:
        policy = fallback

    _POLICY_CACHE[cache_key] = policy
    return policy


def build_policy_from_env() -> BasePolicy:
    return build_policy_from_config(None)






