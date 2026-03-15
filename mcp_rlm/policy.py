from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import asyncio
import json
import os
import re
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
        timeout_seconds: float = 25.0,
        fallback: Optional[BasePolicy] = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
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
                top_k=max(4, min(32, int(result.get("top_k", 12)))),
                coarse_k=max(8, min(64, int(result.get("coarse_k", 24)))),
                max_children=max(4, min(32, int(result.get("max_children", 12)))),
                read_max_chars=max(4000, min(24000, int(result.get("read_max_chars", 12000)))),
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


def build_policy_from_env() -> BasePolicy:
    mode = os.getenv("MCP_RLM_POLICY_MODE", "heuristic").strip().lower()
    if mode != "openai":
        return HeuristicPolicy()

    api_base = os.getenv("MCP_RLM_API_BASE", "").strip()
    model = os.getenv("MCP_RLM_MODEL", "").strip()
    api_key = os.getenv("MCP_RLM_API_KEY", "").strip() or None
    if not api_base or not model:
        return HeuristicPolicy()

    return OpenAICompatiblePolicy(
        api_base=api_base,
        model=model,
        api_key=api_key,
        fallback=HeuristicPolicy(),
    )
