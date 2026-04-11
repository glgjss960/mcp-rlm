from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from time import perf_counter

from .mcp import MCPCall
from .policy import build_policy_from_config
from .types import Budget, GroupResult, MemoryObjectType, WriteReason

if TYPE_CHECKING:
    from .programs import ProgramRegistry
    from .runtime import GroupContext


_DEFAULT_MANAGER_SYSTEM_PROMPT = (
    "You are the root execution manager for MCP-RLM. "
    "You MUST decide the next runtime action as JSON only. "
    "You control tool/object calls, parallel calls, recursive spawn/join, "
    "shared-memory read/write, and finalization.\n\n"
    "Return exactly one JSON object with this schema:\n"
    "{\n"
    '  "type": "call_object | call_many | spawn_groups | join_groups | read_memory | write_memory | flush_context_pressure | finalize",\n'
    '  "rationale": "short reason",\n'
    '  "object_name": "alias/tool",\n'
    '  "payload": {...},\n'
    '  "timeout_seconds": 30,\n'
    '  "calls": [{"object_name": "alias/tool", "payload": {...}, "timeout_seconds": 30}],\n'
    '  "best_effort": false,\n'
    '  "specs": [{"goal": "...", "program": "llm_managed_child", "input_payload": {...}, "budget": {"max_steps": 64, "max_children": 16, "max_wall_seconds": 300, "max_object_calls": 256}}],\n'
    '  "group_ids": ["grp_xxx"],\n'
    '  "join_mode": "explicit | all_pending",\n'
    '  "key": "memory/key",\n'
    '  "object_type": "GOAL | FACT | PLAN | DECISION | ARTIFACT | METRIC",\n'
    '  "reason": "SPAWN_PREP | JOIN_RESULT | CONTEXT_PRESSURE | VALUE_EVENT | FINALIZE",\n'
    '  "content": {...},\n'
    '  "confidence": 0.0,\n'
    '  "force": false,\n'
    '  "note": "optional",\n'
    '  "output": {...}\n'
    "}\n\n"
    "Rules:\n"
    "1) Use call_many when parallelism helps.\n"
    "2) Use spawn_groups for recursive decomposition.\n"
    "3) Use join_groups before final answer if children are pending.\n"
    "4) Persist critical state with write_memory.\n"
    "5) When done, return type=finalize with final output.\n"
    "6) Never return markdown, code fences, or extra text."
)

_DEFAULT_FINALIZE_SYSTEM_PROMPT = (
    "You are finalizing an MCP-RLM episode. "
    "Return ONLY one JSON object as final output for runtime finalize."
)


def _merge_system_prompt(default_prompt: str, task_prompt: str, *, section_title: str) -> str:
    task = str(task_prompt or "").strip()
    if not task:
        return default_prompt
    if task == default_prompt:
        return default_prompt
    return default_prompt + "\n\n" + section_title + "\n" + task

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
    episode_id: str = "",
    group_id: str = "",
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
    if episode_id:
        payload["episode_id"] = episode_id
    if group_id:
        payload["group_id"] = group_id
    if elapsed_ms is not None:
        payload["elapsed_ms"] = int(elapsed_ms)
    if detail:
        payload["detail"] = _jsonable(detail, max_chars=1200)
    print("[mcp-rlm-stage] " + json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


def _clip_text(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f" ... [truncated {len(text) - max_chars} chars]"


def _jsonable(value: Any, *, max_chars: int = 4000) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return _clip_text(value, max_chars=max_chars)
        return value
    if isinstance(value, list):
        return [_jsonable(v, max_chars=max_chars) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v, max_chars=max_chars) for k, v in value.items()}
    return _clip_text(str(value), max_chars=max_chars)


_MCQ_LETTERS = ("A", "B", "C", "D")


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
            letter = str(match.group(1)).strip().upper()
            if letter in _MCQ_LETTERS:
                return letter
    return None


def _normalize_mcq_finalize_output(output: Any, *, input_payload: Dict[str, Any]) -> Any:
    if not isinstance(output, dict):
        return output

    choices_raw = input_payload.get("choices")
    if not isinstance(choices_raw, dict) or not choices_raw:
        return output

    pred = str(output.get("pred", "")).strip().upper()
    if pred not in _MCQ_LETTERS:
        pred = _extract_answer_letter(str(output.get("response", ""))) or ""
    if pred not in _MCQ_LETTERS:
        pred = _extract_answer_letter(str(output.get("answer", ""))) or ""

    if pred not in _MCQ_LETTERS:
        # fallback to first available option for deterministic eval format
        ordered = [str(k).strip().upper() for k in choices_raw.keys()]
        ordered = [k for k in ordered if k in _MCQ_LETTERS]
        if ordered:
            pred = ordered[0]

    if pred in _MCQ_LETTERS:
        output["pred"] = pred
        if not str(output.get("response", "")).strip():
            output["response"] = f"The correct answer is ({pred})"

    return output


def _extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be object")
    return payload


async def _policy_chat_json(
    policy: Any,
    *,
    system: str,
    user: str,
    timeout_seconds: float,
    stage_log_enabled: bool,
    episode_id: str,
    group_id: str,
    stage_label: str,
) -> Dict[str, Any]:
    chat_json = getattr(policy, "_chat_json", None)
    if callable(chat_json):
        start = perf_counter()
        _stage_log(
            enabled=stage_log_enabled,
            component="llm_manager",
            stage=stage_label,
            status="start",
            episode_id=episode_id,
            group_id=group_id,
            detail={
                "timeout_seconds": timeout_seconds,
                "policy_type": type(policy).__name__,
                "user_chars": len(user),
            },
        )
        try:
            raw = await asyncio.wait_for(
                chat_json(system=system, user=user),
                timeout=_to_timeout(timeout_seconds, default=120.0),
            )
        except asyncio.TimeoutError as exc:
            elapsed_ms = int((perf_counter() - start) * 1000)
            _stage_log(
                enabled=stage_log_enabled,
                component="llm_manager",
                stage=stage_label,
                status="timeout",
                episode_id=episode_id,
                group_id=group_id,
                elapsed_ms=elapsed_ms,
                detail={"timeout_seconds": timeout_seconds},
            )
            raise RuntimeError(
                f"{stage_label} timed out after {timeout_seconds:.1f}s (policy={type(policy).__name__})"
            ) from exc
        except Exception as exc:
            elapsed_ms = int((perf_counter() - start) * 1000)
            _stage_log(
                enabled=stage_log_enabled,
                component="llm_manager",
                stage=stage_label,
                status="error",
                episode_id=episode_id,
                group_id=group_id,
                elapsed_ms=elapsed_ms,
                detail={"error": str(exc), "error_type": type(exc).__name__},
            )
            raise

        elapsed_ms = int((perf_counter() - start) * 1000)
        _stage_log(
            enabled=stage_log_enabled,
            component="llm_manager",
            stage=stage_label,
            status="ok",
            episode_id=episode_id,
            group_id=group_id,
            elapsed_ms=elapsed_ms,
        )
        if isinstance(raw, dict):
            return raw
        return _extract_json_object(str(raw))
    raise RuntimeError(
        "Selected policy mode does not support manager JSON action loop. "
        "Use policy-mode openai/openrouter/vllm/ollama/huggingface."
    )


def _parse_budget(raw: Any) -> Optional[Budget]:
    if not isinstance(raw, dict):
        return None
    try:
        return Budget(
            max_steps=max(1, int(raw.get("max_steps", 64))),
            max_children=max(1, int(raw.get("max_children", 32))),
            max_wall_seconds=max(1.0, float(raw.get("max_wall_seconds", 120.0))),
            max_object_calls=max(1, int(raw.get("max_object_calls", 256))),
        )
    except Exception:
        return None


def _parse_object_type(raw: Any) -> MemoryObjectType:
    value = str(raw or "ARTIFACT").strip().upper()
    try:
        return MemoryObjectType(value)
    except Exception:
        return MemoryObjectType.ARTIFACT


def _parse_write_reason(raw: Any) -> WriteReason:
    value = str(raw or "VALUE_EVENT").strip().upper()
    try:
        return WriteReason(value)
    except Exception:
        return WriteReason.VALUE_EVENT


def _format_group_result(item: GroupResult) -> Dict[str, Any]:
    return {
        "group_id": item.group_id,
        "status": item.status.value,
        "reward": item.reward,
        "error": item.error,
        "child_group_ids": list(item.child_group_ids),
        "output": _jsonable(item.output, max_chars=2400),
    }


async def _list_available_objects(
    ctx: "GroupContext",
    *,
    timeout_seconds: float,
    stage_log_enabled: bool,
) -> List[str]:
    client = getattr(ctx._runtime, "mcp_client", None)
    if client is None:
        return []
    list_fn = getattr(client, "list_objects", None)
    if not callable(list_fn):
        return []
    start = perf_counter()
    _stage_log(
        enabled=stage_log_enabled,
        component="llm_manager",
        stage="list_objects",
        status="start",
        episode_id=ctx.episode_id,
        group_id=ctx.group_id,
        detail={"timeout_seconds": timeout_seconds},
    )
    try:
        try:
            items = await asyncio.wait_for(
                list_fn(timeout_seconds=timeout_seconds),
                timeout=_to_timeout(timeout_seconds, default=20.0),
            )
        except TypeError:
            items = await asyncio.wait_for(
                list_fn(),
                timeout=_to_timeout(timeout_seconds, default=20.0),
            )
    except asyncio.TimeoutError:
        _stage_log(
            enabled=stage_log_enabled,
            component="llm_manager",
            stage="list_objects",
            status="timeout",
            episode_id=ctx.episode_id,
            group_id=ctx.group_id,
            elapsed_ms=int((perf_counter() - start) * 1000),
            detail={"timeout_seconds": timeout_seconds},
        )
        return []
    except Exception as exc:
        _stage_log(
            enabled=stage_log_enabled,
            component="llm_manager",
            stage="list_objects",
            status="error",
            episode_id=ctx.episode_id,
            group_id=ctx.group_id,
            elapsed_ms=int((perf_counter() - start) * 1000),
            detail={"error": str(exc), "error_type": type(exc).__name__},
        )
        return []
    out = sorted([str(x) for x in items if str(x).strip()])
    _stage_log(
        enabled=stage_log_enabled,
        component="llm_manager",
        stage="list_objects",
        status="ok",
        episode_id=ctx.episode_id,
        group_id=ctx.group_id,
        elapsed_ms=int((perf_counter() - start) * 1000),
        detail={"num_objects": len(out)},
    )
    return out


def _pending_status_snapshot(ctx: "GroupContext", pending_ids: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    groups = getattr(ctx._runtime, "groups", {})
    for gid in pending_ids:
        group = groups.get(gid)
        if group is None:
            out.append({"group_id": gid, "status": "UNKNOWN"})
            continue
        out.append(
            {
                "group_id": gid,
                "status": group.status.value,
                "goal": str(group.goal),
                "program": str(group.program),
                "error": group.error,
            }
        )
    return out


def _compact_history(history: List[Dict[str, Any]], *, max_items: int) -> List[Dict[str, Any]]:
    if max_items <= 0:
        return []
    return history[-max_items:]


def _build_turn_payload(
    ctx: "GroupContext",
    *,
    turn_index: int,
    max_turns: int,
    available_objects: List[str],
    pending_children: List[str],
    recent_history: List[Dict[str, Any]],
    last_observation: Dict[str, Any],
    known_memory_keys: List[str],
) -> Dict[str, Any]:
    return {
        "turn_index": turn_index,
        "max_turns": max_turns,
        "goal": ctx.goal,
        "input_payload": _jsonable(ctx.input_payload, max_chars=2600),
        "context_usage": round(float(ctx.context_usage()), 4),
        "available_objects": available_objects,
        "pending_children": list(pending_children),
        "pending_children_status": _pending_status_snapshot(ctx, pending_children),
        "known_memory_keys": list(known_memory_keys),
        "recent_history": _compact_history(recent_history, max_items=8),
        "last_observation": _jsonable(last_observation, max_chars=2400),
    }


async def _execute_action(
    ctx: "GroupContext",
    *,
    action: Dict[str, Any],
    pending_children: List[str],
    known_memory_keys: List[str],
    default_child_program: str,
) -> tuple[Dict[str, Any], bool, Optional[Any]]:
    action_type = str(action.get("type") or action.get("action_type") or "").strip().lower()
    if not action_type:
        raise RuntimeError("Manager action missing type")

    if action_type == "call_object":
        object_name = str(action.get("object_name", "")).strip()
        if not object_name:
            raise RuntimeError("call_object requires object_name")
        payload = action.get("payload", {})
        if not isinstance(payload, dict):
            payload = {"value": payload}
        timeout_seconds = float(action.get("timeout_seconds", 30.0))
        output = await ctx.call_object(object_name, payload, timeout_seconds=timeout_seconds)
        obs = {
            "type": "call_object",
            "object_name": object_name,
            "ok": True,
            "output": _jsonable(output, max_chars=2200),
        }
        return obs, False, None

    if action_type in {"call_many", "call_objects"}:
        calls_raw = action.get("calls", [])
        if not isinstance(calls_raw, list) or not calls_raw:
            raise RuntimeError("call_many requires non-empty calls list")

        calls: List[MCPCall] = []
        for item in calls_raw:
            if not isinstance(item, dict):
                continue
            object_name = str(item.get("object_name", "")).strip()
            if not object_name:
                continue
            payload = item.get("payload", {})
            if not isinstance(payload, dict):
                payload = {"value": payload}
            timeout_seconds = float(item.get("timeout_seconds", action.get("timeout_seconds", 30.0)))
            calls.append(MCPCall(object_name=object_name, payload=payload, timeout_seconds=timeout_seconds))

        if not calls:
            raise RuntimeError("call_many has no valid call specs")

        best_effort = bool(action.get("best_effort", False))
        results: List[Dict[str, Any]] = []
        if best_effort:
            async def one(call: MCPCall) -> Dict[str, Any]:
                try:
                    output = await ctx.call_object(call.object_name, call.payload, timeout_seconds=call.timeout_seconds)
                    return {
                        "object_name": call.object_name,
                        "ok": True,
                        "output": _jsonable(output, max_chars=1600),
                    }
                except Exception as exc:
                    return {
                        "object_name": call.object_name,
                        "ok": False,
                        "error": str(exc),
                    }

            tasks = [asyncio.create_task(one(call)) for call in calls]
            results = await asyncio.gather(*tasks)
        else:
            outputs = await ctx.call_objects(calls)
            for idx, call in enumerate(calls):
                output = outputs[idx] if idx < len(outputs) else None
                results.append(
                    {
                        "object_name": call.object_name,
                        "ok": True,
                        "output": _jsonable(output, max_chars=1600),
                    }
                )

        obs = {
            "type": "call_many",
            "num_calls": len(calls),
            "best_effort": best_effort,
            "results": results,
        }
        return obs, False, None

    if action_type in {"spawn_groups", "spawn"}:
        specs_raw = action.get("specs") or action.get("children") or []
        if not isinstance(specs_raw, list) or not specs_raw:
            raise RuntimeError("spawn_groups requires specs list")

        inherit_keys = [
            "query",
            "question",
            "choices",
            "manifest_path",
            "policy_config",
            "manager_max_turns",
            "manager_max_history",
            "default_child_program",
            "manager_system_prompt",
            "manager_finalize_system_prompt",
        ]

        specs: List[Dict[str, Any]] = []
        for item in specs_raw:
            if not isinstance(item, dict):
                continue
            payload = item.get("input_payload", {})
            if not isinstance(payload, dict):
                payload = {}

            for key in inherit_keys:
                if key not in payload and key in ctx.input_payload:
                    payload[key] = ctx.input_payload[key]

            child_program = str(item.get("program", "")).strip() or default_child_program
            goal = str(item.get("goal", "")).strip() or f"Sub-goal {len(specs) + 1}"

            spec: Dict[str, Any] = {
                "goal": goal,
                "program": child_program,
                "input_payload": payload,
            }
            budget = _parse_budget(item.get("budget"))
            if budget is not None:
                spec["budget"] = budget
            specs.append(spec)

        if not specs:
            raise RuntimeError("spawn_groups has no valid specs")

        child_ids = await ctx.spawn_groups(specs)
        for gid in child_ids:
            if gid not in pending_children:
                pending_children.append(gid)

        obs = {
            "type": "spawn_groups",
            "spawned_group_ids": list(child_ids),
            "num_spawned": len(child_ids),
            "pending_children": list(pending_children),
        }
        return obs, False, None

    if action_type in {"join_groups", "join"}:
        join_mode = str(action.get("join_mode", "explicit")).strip().lower()
        group_ids_raw = action.get("group_ids", [])
        group_ids: List[str] = []

        if join_mode == "all_pending":
            group_ids = list(pending_children)
        elif isinstance(group_ids_raw, list):
            group_ids = [str(x).strip() for x in group_ids_raw if str(x).strip()]

        if not group_ids:
            return {
                "type": "join_groups",
                "num_joined": 0,
                "results": [],
                "pending_children": list(pending_children),
            }, False, None

        results = await ctx.join_groups(group_ids)
        pending_children[:] = [gid for gid in pending_children if gid not in set(group_ids)]

        obs = {
            "type": "join_groups",
            "num_joined": len(group_ids),
            "results": [_format_group_result(item) for item in results],
            "pending_children": list(pending_children),
        }
        return obs, False, None

    if action_type == "read_memory":
        key = str(action.get("key", "")).strip()
        if not key:
            raise RuntimeError("read_memory requires key")
        result = await ctx.read_memory(key)
        if key not in known_memory_keys:
            known_memory_keys.append(key)
        obs = {
            "type": "read_memory",
            "key": key,
            "result": _jsonable(result, max_chars=2200),
        }
        return obs, False, None

    if action_type == "write_memory":
        key = str(action.get("key", "")).strip()
        if not key:
            raise RuntimeError("write_memory requires key")
        object_type = _parse_object_type(action.get("object_type"))
        reason = _parse_write_reason(action.get("reason"))
        content = action.get("content", {})
        confidence = float(action.get("confidence", 1.0))
        force = bool(action.get("force", False))

        event = await ctx.write_memory(
            key=key,
            object_type=object_type,
            reason=reason,
            content=content,
            confidence=confidence,
            force=force,
        )
        if key not in known_memory_keys:
            known_memory_keys.append(key)

        obs = {
            "type": "write_memory",
            "key": key,
            "object_type": object_type.value,
            "reason": reason.value,
            "event": _jsonable(event, max_chars=1200),
        }
        return obs, False, None

    if action_type == "flush_context_pressure":
        note = str(action.get("note", "")).strip()
        force = bool(action.get("force", False))
        event = await ctx.flush_context_pressure(note=note, force=force)
        obs = {
            "type": "flush_context_pressure",
            "note": note,
            "event": _jsonable(event, max_chars=1200),
        }
        return obs, False, None

    if action_type == "finalize":
        output = _normalize_mcq_finalize_output(action.get("output", {}), input_payload=ctx.input_payload)
        finalized = await ctx.finalize(output)
        obs = {
            "type": "finalize",
            "output": _jsonable(finalized, max_chars=2200),
        }
        return obs, True, finalized

    raise RuntimeError(f"Unsupported manager action type: {action_type}")


async def llm_managed_group_program(ctx: "GroupContext") -> Dict[str, Any]:
    policy_config = ctx.input_payload.get("policy_config")
    if not isinstance(policy_config, dict):
        policy_config = {}

    policy = build_policy_from_config(policy_config)

    max_turns = max(1, int(ctx.input_payload.get("manager_max_turns", 48)))
    max_history = max(1, int(ctx.input_payload.get("manager_max_history", 8)))
    default_child_program = str(ctx.input_payload.get("default_child_program", "llm_managed_child")).strip() or "llm_managed_child"

    manager_system_prompt = _merge_system_prompt(
        _DEFAULT_MANAGER_SYSTEM_PROMPT,
        str(ctx.input_payload.get("manager_system_prompt", "")),
        section_title="Task-specific policy:",
    )
    finalize_system_prompt = _merge_system_prompt(
        _DEFAULT_FINALIZE_SYSTEM_PROMPT,
        str(ctx.input_payload.get("manager_finalize_system_prompt", "")),
        section_title="Task-specific finalize policy:",
    )
    stage_log_enabled = _to_bool(
        ctx.input_payload.get("manager_debug_stage_logs"),
        default=_to_bool(os.getenv("MCP_RLM_DEBUG_STAGE_LOGS"), default=True),
    )
    list_objects_timeout_seconds = _to_timeout(
        ctx.input_payload.get(
            "manager_list_objects_timeout_seconds",
            os.getenv("MCP_RLM_MANAGER_LIST_OBJECTS_TIMEOUT_SECONDS", "20"),
        ),
        default=20.0,
    )
    policy_chat_timeout_seconds = _to_timeout(
        ctx.input_payload.get(
            "manager_policy_chat_timeout_seconds",
            os.getenv("MCP_RLM_MANAGER_POLICY_CHAT_TIMEOUT_SECONDS", "120"),
        ),
        default=120.0,
    )

    script = ctx.input_payload.get("manager_script", [])
    if not isinstance(script, list):
        script = []
    script_index = 0

    available_objects = await _list_available_objects(
        ctx,
        timeout_seconds=list_objects_timeout_seconds,
        stage_log_enabled=stage_log_enabled,
    )
    pending_children: List[str] = []
    known_memory_keys: List[str] = []
    history: List[Dict[str, Any]] = []
    last_observation: Dict[str, Any] = {
        "type": "bootstrap",
        "note": "Manager loop started",
    }

    for turn in range(1, max_turns + 1):
        turn_payload = _build_turn_payload(
            ctx,
            turn_index=turn,
            max_turns=max_turns,
            available_objects=available_objects,
            pending_children=pending_children,
            recent_history=history,
            last_observation=last_observation,
            known_memory_keys=known_memory_keys,
        )

        if script_index < len(script) and isinstance(script[script_index], dict):
            action = dict(script[script_index])
            script_index += 1
        else:
            action_raw = await _policy_chat_json(
                policy,
                system=manager_system_prompt,
                user=json.dumps(turn_payload, ensure_ascii=False),
                timeout_seconds=policy_chat_timeout_seconds,
                stage_log_enabled=stage_log_enabled,
                episode_id=ctx.episode_id,
                group_id=ctx.group_id,
                stage_label=f"manager_turn_{turn}_policy_chat",
            )
            action = action_raw.get("action", action_raw) if isinstance(action_raw, dict) else {}
            if not isinstance(action, dict):
                raise RuntimeError("Manager policy returned non-dict action")

        action_summary = {
            "type": str(action.get("type") or action.get("action_type") or ""),
            "rationale": str(action.get("rationale", "")),
            "object_name": str(action.get("object_name", "")),
            "join_mode": str(action.get("join_mode", "")),
        }

        try:
            observation, done, final_output = await _execute_action(
                ctx,
                action=action,
                pending_children=pending_children,
                known_memory_keys=known_memory_keys,
                default_child_program=default_child_program,
            )
        except Exception as exc:
            observation = {
                "type": "action_error",
                "error": str(exc),
                "failed_action": _jsonable(action, max_chars=1800),
            }
            done = False
            final_output = None

        history.append(
            {
                "turn": turn,
                "action": _jsonable(action_summary, max_chars=1000),
                "action_full": _jsonable(action, max_chars=2200),
                "observation": _jsonable(observation, max_chars=2000),
            }
        )
        if len(history) > max_history:
            history = history[-max_history:]

        last_observation = observation

        if done:
            if isinstance(final_output, dict):
                final_output.setdefault("manager_turns", turn)
                final_output.setdefault("manager_history", history)
            return final_output if isinstance(final_output, dict) else {
                "output": final_output,
                "manager_turns": turn,
                "manager_history": history,
            }

    if pending_children:
        try:
            join_results = await ctx.join_groups(list(pending_children))
            last_observation = {
                "type": "auto_join_after_max_turns",
                "results": [_format_group_result(item) for item in join_results],
            }
            pending_children.clear()
        except Exception as exc:
            last_observation = {
                "type": "auto_join_after_max_turns_error",
                "error": str(exc),
            }

    finalize_payload = {
        "goal": ctx.goal,
        "input_payload": _jsonable(ctx.input_payload, max_chars=2200),
        "history": history,
        "last_observation": _jsonable(last_observation, max_chars=2200),
        "context_usage": round(float(ctx.context_usage()), 4),
        "pending_children": pending_children,
    }

    try:
        final_raw = await _policy_chat_json(
            policy,
            system=finalize_system_prompt,
            user=json.dumps(finalize_payload, ensure_ascii=False),
            timeout_seconds=policy_chat_timeout_seconds,
            stage_log_enabled=stage_log_enabled,
            episode_id=ctx.episode_id,
            group_id=ctx.group_id,
            stage_label="manager_finalize_policy_chat",
        )
        final_output = final_raw.get("output", final_raw) if isinstance(final_raw, dict) else {"output": final_raw}
        if not isinstance(final_output, dict):
            final_output = {"output": final_output}
    except Exception as exc:
        final_output = {
            "error": "Manager reached max turns and finalize policy failed",
            "finalize_error": str(exc),
            "last_observation": _jsonable(last_observation, max_chars=2200),
            "pending_children": list(pending_children),
        }

    final_output = _normalize_mcq_finalize_output(final_output, input_payload=ctx.input_payload)
    final_output.setdefault("manager_max_turns_reached", True)
    final_output.setdefault("manager_turns", max_turns)
    final_output.setdefault("manager_history", history)
    return await ctx.finalize(final_output)


def register_llm_manager_programs(registry: "ProgramRegistry") -> None:
    registry.register("llm_managed_root", llm_managed_group_program)
    registry.register("llm_managed_child", llm_managed_group_program)

