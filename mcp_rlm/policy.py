from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import asyncio
import json
import os
import urllib.request


def _extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


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
        api_base: str,
        model: str,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 25.0,
        fallback: Optional[BasePolicy] = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
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

    async def _chat_json(self, *, system: str, user: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system + " Return ONLY valid JSON object."},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }

        def _request() -> Dict[str, Any]:
            data = json.dumps(payload).encode("utf-8")
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
            return _extract_json_object(content)

        return await asyncio.to_thread(_request)


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
        fallback: Optional[BasePolicy] = None,
    ) -> None:
        self.model = model
        self.revision = revision
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.fallback = fallback or HeuristicPolicy()

        self._pipeline: Any = None
        self._load_lock = asyncio.Lock()

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

    async def _chat_json(self, *, system: str, user: str) -> Dict[str, Any]:
        await self._ensure_loaded()

        prompt = (
            "System:\n"
            + system
            + "\n\nUser:\n"
            + user
            + "\n\nReturn ONLY valid JSON object.\nAssistant:\n"
        )

        def _infer() -> Dict[str, Any]:
            if self._pipeline is None:
                raise RuntimeError("Transformers pipeline is not initialized")
            try:
                outputs = self._pipeline(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    return_full_text=False,
                )
            except TypeError:
                outputs = self._pipeline(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
            if not outputs:
                raise RuntimeError("Empty generation output")
            first = outputs[0]
            text = str(first.get("generated_text", "")) if isinstance(first, dict) else str(first)
            if not text and isinstance(first, dict):
                text = str(first.get("text", ""))
            if text.startswith(prompt):
                text = text[len(prompt) :]
            return _extract_json_object(text)

        return await asyncio.to_thread(_infer)

    async def _ensure_loaded(self) -> None:
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
                api_base=api_base,
                model=model,
                api_key=api_key,
                extra_headers=headers,
                fallback=fallback,
            )
    elif mode in {"huggingface", "hf", "transformers"}:
        model = str(resolved.get("model", "")).strip()
        if not model:
            policy = fallback
        else:
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
                fallback=fallback,
            )
    else:
        policy = fallback

    _POLICY_CACHE[cache_key] = policy
    return policy


def build_policy_from_env() -> BasePolicy:
    return build_policy_from_config(None)





